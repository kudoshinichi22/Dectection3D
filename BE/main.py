import logging
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from camera_service import CameraService
from detector import ObjectDetector
from pose_3d import bbox_to_pseudo_3d
from pose_estimator import PoseEstimator
from tracker import SimpleObjectTracker


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="YOLO 2D Detection + Pose + Pseudo-3D")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = ObjectDetector()
pose_estimator = PoseEstimator()
camera_service = CameraService()


def read_upload_image(file_content: bytes):
    np_arr = np.frombuffer(file_content, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def enrich_objects(frame, frame_id: int, source_type: str, tracker: SimpleObjectTracker, timestamp: float | None = None) -> list[dict]:
    height, width = frame.shape[:2]
    objects = []

    for obj in detector.detect(frame):
        obj["source"] = "detect"
        obj["position_3d"] = bbox_to_pseudo_3d(
            obj["bbox_2d"],
            width,
            height,
            obj.get("confidence", 1.0),
        )
        objects.append(obj)

    for pose in pose_estimator.estimate(frame):
        pose["source"] = "pose"
        pose["position_3d"] = bbox_to_pseudo_3d(
            pose["bbox_2d"],
            width,
            height,
            pose.get("confidence", 1.0),
        )
        objects.append(pose)

    objects = tracker.update(objects, frame_id, timestamp)
    return objects


def response_payload(source_type: str, frame_id: int, frame, objects: list[dict], **extra) -> dict:
    height, width = frame.shape[:2]
    payload = {
        "success": True,
        "source_type": source_type,
        "frame_id": frame_id,
        "image_size": {"width": width, "height": height},
        "model_status": {
            "detector": detector.info,
            "pose": pose_estimator.info,
        },
        "objects": objects,
        "detections": objects,
        "total_objects": len(objects),
    }
    payload.update(extra)
    return payload


@app.get("/")
def home():
    return {
        "success": True,
        "message": "Backend is running",
        "model_status": {
            "detector": detector.info,
            "pose": pose_estimator.info,
        },
    }


@app.get("/model/status")
def model_status():
    return {
        "success": True,
        "detector": detector.info,
        "pose": pose_estimator.info,
        "pseudo_3d": {
            "method": "pseudo_3d",
            "note": "No metric depth is claimed without calibration/depth input.",
        },
    }


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    image = read_upload_image(await file.read())
    if image is None:
        return {"success": False, "error": "Không đọc được ảnh"}

    tracker = SimpleObjectTracker()
    objects = enrich_objects(image, frame_id=0, source_type="image", tracker=tracker, timestamp=time.time())

    LOGGER.info("Image detection: %s, objects=%s", file.filename, len(objects))
    for obj in objects:
        LOGGER.info(
            "Object %s | %s | conf=%s | bbox=%s | position_3d=%s",
            obj.get("object_id"),
            obj.get("class_name"),
            obj.get("confidence"),
            obj.get("bbox_2d"),
            obj.get("position_3d"),
        )

    return response_payload("image", 0, image, objects, filename=file.filename)


# Serve frontend files (index.html, app.js, style.css)
# Mount LAST so it doesn't shadow API routes
try:
    from pathlib import Path as _Path
    _FE_DIR = _Path(__file__).resolve().parent
    if (_FE_DIR / "index.html").exists():
        from fastapi.staticfiles import StaticFiles as _SF
        app.mount("/static", _SF(directory=str(_FE_DIR), name="frontend"), name="frontend_static")
        from fastapi.responses import FileResponse as _FR
        @app.get("/ui")
        def serve_frontend():
            return _FR(str(_FE_DIR / "index.html"))
except Exception as _e:
    LOGGER.warning("Could not mount frontend: %s", _e)


@app.get("/camera/frame")
def camera_frame():
    try:
        return camera_service.process_next_frame(detector, pose_estimator)
    except RuntimeError as exc:
        return {"success": False, "source_type": "camera", "error": str(exc)}
