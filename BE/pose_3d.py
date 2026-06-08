from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

COCO_PERSON_SKELETON = [
    [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 6], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16],
    [0, 1], [0, 2], [1, 3], [2, 4],
]


def load_camera_calibration(path: str | Path | None = None) -> dict:
    calibration_path = Path(path) if path else BASE_DIR / "camera_calibration.json"
    if not calibration_path.exists():
        return {"available": False}

    import json

    with calibration_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    data["available"] = True
    return data


def bbox_to_pseudo_3d(bbox_2d: list[float], image_width: int, image_height: int, confidence: float = 1.0) -> dict:
    x1, y1, x2, y2 = bbox_2d
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    box_width = max(x2 - x1, 1.0)
    box_height = max(y2 - y1, 1.0)

    x = (center_x - image_width / 2) / (image_width / 2)
    y = -((center_y - image_height / 2) / (image_height / 2))
    scale = max(box_width / image_width, box_height / image_height, 0.01)
    z = 1.0 / scale

    return {
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
        "scale": round(scale, 4),
        "method": "pseudo_3d",
        "confidence": round(float(confidence), 3),
        "note": "Pseudo-3D from 2D bbox scale. This is not metric 3D depth.",
    }


def keypoints_to_pseudo_3d(keypoints_2d: list[list[float]], image_width: int, image_height: int) -> list[dict]:
    points_3d = []
    visible_points = [point for point in keypoints_2d if len(point) >= 3 and point[2] > 0.2]

    if not visible_points:
        return points_3d

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]
    spread = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    scale = max(spread / max(image_width, image_height), 0.01)
    z = 1.0 / scale

    for index, point in enumerate(keypoints_2d):
        if len(point) < 3:
            continue
        x, y, confidence = point
        points_3d.append({
            "index": index,
            "x": round((x - image_width / 2) / (image_width / 2), 4),
            "y": round(-((y - image_height / 2) / (image_height / 2)), 4),
            "z": round(z, 4),
            "confidence": round(float(confidence), 3),
            "method": "pseudo_3d",
        })

    return points_3d


def pose_center(keypoints_2d: list[list[float]]) -> dict | None:
    visible_points = [point for point in keypoints_2d if len(point) >= 3 and point[2] > 0.2]
    if not visible_points:
        return None
    return {
        "x": round(sum(point[0] for point in visible_points) / len(visible_points), 2),
        "y": round(sum(point[1] for point in visible_points) / len(visible_points), 2),
        "confidence": round(sum(point[2] for point in visible_points) / len(visible_points), 3),
    }
