import logging
from pathlib import Path

from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
MIN_MODEL_BYTES = 1024 * 1024


def is_valid_model_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES


def resolve_model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    return model_path


def select_detection_model() -> tuple[Path, str]:
    candidates = [
        (BASE_DIR.parent / "runs" / "detect" / "train" / "weights" / "best.pt", "custom"),
        (BASE_DIR / "models" / "best.pt", "custom"),
        (BASE_DIR / "yolov8n.pt", "default"),
    ]

    for model_path, model_type in candidates:
        if is_valid_model_file(model_path):
            return model_path, model_type
        if model_path.exists():
            LOGGER.warning("Ignoring invalid or empty model file: %s", model_path)

    return BASE_DIR / "yolov8n.pt", "default"


class ObjectDetector:
    def __init__(self, model_path: str | Path | None = None):
        if model_path:
            selected_path = resolve_model_path(model_path)
            model_type = "custom" if "best.pt" in selected_path.name else "default"
            if not is_valid_model_file(selected_path):
                LOGGER.warning("Requested model is missing or invalid: %s", selected_path)
                selected_path, model_type = select_detection_model()
        else:
            selected_path, model_type = select_detection_model()

        self.model_path = selected_path
        self.model_type = model_type
        self.model = YOLO(str(selected_path))
        LOGGER.info("Loaded %s YOLO detection model: %s", model_type, selected_path)

    @property
    def info(self) -> dict:
        return {
            "task": "detect",
            "type": self.model_type,
            "path": str(self.model_path),
            "valid": is_valid_model_file(self.model_path),
        }

    def detect(self, image) -> list[dict]:
        results = self.model(image, conf=0.25, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox_2d": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ],
                })

        return detections
