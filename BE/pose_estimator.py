import logging
from pathlib import Path

from ultralytics import YOLO

from pose_3d import COCO_PERSON_SKELETON, keypoints_to_pseudo_3d, pose_center


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
MIN_MODEL_BYTES = 1024 * 1024


class PoseEstimator:
    def __init__(self, model_path: str | Path = "yolov8n-pose.pt"):
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            local_path = BASE_DIR / self.model_path
            self.model_path = local_path if local_path.exists() else self.model_path

        self.model = None
        self.enabled = False
        self.error = None

        try:
            self.model = YOLO(str(self.model_path))
            self.enabled = True
            LOGGER.info("Loaded YOLO pose model: %s", self.model_path)
        except Exception as exc:
            self.error = str(exc)
            LOGGER.warning("Pose model is not available: %s", exc)

    @property
    def info(self) -> dict:
        return {
            "task": "pose",
            "type": "yolo_pose",
            "path": str(self.model_path),
            "enabled": self.enabled,
            "error": self.error,
            "note": "YOLOv8 pose default model is trained mainly for human pose.",
        }

    def estimate(self, image) -> list[dict]:
        if not self.enabled or self.model is None:
            return []

        height, width = image.shape[:2]
        results = self.model(image, conf=0.25, verbose=False)
        poses = []

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is None or keypoints is None:
                continue

            for index, box in enumerate(boxes):
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                points = []

                if keypoints.data is not None and index < len(keypoints.data):
                    for point in keypoints.data[index].tolist():
                        if len(point) >= 3:
                            points.append([
                                round(float(point[0]), 2),
                                round(float(point[1]), 2),
                                round(float(point[2]), 3),
                            ])

                poses.append({
                    "class_name": "person",
                    "confidence": round(confidence, 3),
                    "bbox_2d": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                    "keypoints_2d": points,
                    "keypoints_3d": keypoints_to_pseudo_3d(points, width, height),
                    "skeleton": COCO_PERSON_SKELETON,
                    "pose_center": pose_center(points),
                })

        return poses


def bbox_2d_to_3d(bbox, image_width, image_height):
    from pose_3d import bbox_to_pseudo_3d

    if isinstance(bbox, dict):
        bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
    return bbox_to_pseudo_3d(bbox, image_width, image_height)
