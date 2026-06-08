import json
import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from camera_service import CameraService
from detector import ObjectDetector
from pose_3d import bbox_to_pseudo_3d
from pose_estimator import PoseEstimator
from tracker import SimpleObjectTracker


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_videos"
OUTPUT_DIR = BASE_DIR / "output_videos"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="YOLO 2D Detection + Pose + Pseudo-3D")

app.mount("/uploaded_videos", StaticFiles(directory=str(UPLOAD_DIR)), name="uploaded_videos")
app.mount("/output_videos", StaticFiles(directory=str(OUTPUT_DIR)), name="output_videos")

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


def ndjson_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


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


def draw_objects_on_frame(frame, objects: list[dict]):
    for obj in objects:
        x1, y1, x2, y2 = [int(value) for value in obj["bbox_2d"]]
        label = f"{obj.get('object_id', '?')} {obj['class_name']} {obj['confidence']}"
        color = (0, 255, 0) if obj.get("source") != "pose" else (255, 180, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        keypoints = obj.get("keypoints_2d") or []
        for x, y, confidence in keypoints:
            if confidence > 0.2:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)

        for start, end in obj.get("skeleton") or []:
            if start < len(keypoints) and end < len(keypoints):
                p1 = keypoints[start]
                p2 = keypoints[end]
                if p1[2] > 0.2 and p2[2] > 0.2:
                    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 200, 255), 2)

    return frame


def create_browser_video_writer(width: int, height: int, fps: float):
    codec_options = [
        (".webm", "VP80"),
        (".webm", "VP90"),
        (".mp4", "avc1"),
        (".mp4", "H264"),
        (".mp4", "mp4v"),
    ]

    for extension, codec in codec_options:
        output_filename = f"detected_{uuid.uuid4()}{extension}"
        output_path = OUTPUT_DIR / output_filename
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, output_filename
        writer.release()

    return None, None


def build_video_summary(class_frame_counts: dict[str, int], processed_frames: int) -> list[dict]:
    if processed_frames <= 0:
        return []

    return [
        {
            "class_name": class_name,
            "frame_count": frame_count,
            "appearance_percent": round(frame_count * 100 / processed_frames, 2),
        }
        for class_name, frame_count in sorted(class_frame_counts.items(), key=lambda item: (-item[1], item[0]))
    ]


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


@app.post("/detect/video/stream")
async def detect_video_stream(file: UploadFile = File(...)):
    file_ext = Path(file.filename or "").suffix or ".mp4"
    input_filename = f"{uuid.uuid4()}{file_ext}"
    input_path = UPLOAD_DIR / input_filename
    input_path.write_bytes(await file.read())

    def stream_detections():
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            yield ndjson_line({"success": False, "type": "error", "error": "Không mở được video"})
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = SimpleObjectTracker()
        class_frame_counts = {}

        yield ndjson_line({
            "success": True,
            "type": "metadata",
            "source_type": "video",
            "filename": file.filename,
            "fps": fps,
            "total_frames": total_frames,
            "video_size": {"width": width, "height": height},
            "model_status": {"detector": detector.info, "pose": pose_estimator.info},
        })

        frame_id = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_id / fps if fps > 0 else None
                objects = enrich_objects(frame, frame_id, "video", tracker, timestamp)

                for class_name in {obj.get("class_name") for obj in objects if obj.get("class_name")}:
                    class_frame_counts[class_name] = class_frame_counts.get(class_name, 0) + 1

                yield ndjson_line({
                    "success": True,
                    "type": "frame",
                    "source_type": "video",
                    "frame_id": frame_id,
                    "frame_index": frame_id,
                    "objects": objects,
                    "detections": objects,
                    "processed_frames": frame_id + 1,
                    "summary": build_video_summary(class_frame_counts, frame_id + 1),
                })
                frame_id += 1
        finally:
            cap.release()

        yield ndjson_line({
            "success": True,
            "type": "done",
            "source_type": "video",
            "processed_frames": frame_id,
            "summary": build_video_summary(class_frame_counts, frame_id),
        })

    return StreamingResponse(stream_detections(), media_type="application/x-ndjson")


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    file_ext = Path(file.filename or "").suffix or ".mp4"
    input_filename = f"{uuid.uuid4()}{file_ext}"
    input_path = UPLOAD_DIR / input_filename
    input_path.write_bytes(await file.read())

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return {"success": False, "error": "Không mở được video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, output_filename = create_browser_video_writer(width, height, fps)
    if writer is None:
        cap.release()
        return {"success": False, "error": "Không tạo được video output"}

    tracker = SimpleObjectTracker()
    results = []
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            objects = enrich_objects(frame, frame_id, "video", tracker, frame_id / fps if fps > 0 else None)
            results.append({"frame_id": frame_id, "objects": objects, "detections": objects})
            writer.write(draw_objects_on_frame(frame, objects))
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    return {
        "success": True,
        "source_type": "video",
        "filename": file.filename,
        "input_video_url": f"http://127.0.0.1:8000/uploaded_videos/{input_filename}",
        "annotated_video_url": f"http://127.0.0.1:8000/output_videos/{output_filename}",
        "total_frames": frame_id,
        "processed_frames": frame_id,
        "video_size": {"width": width, "height": height},
        "fps": fps,
        "model_status": {"detector": detector.info, "pose": pose_estimator.info},
        "results": results,
    }


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
