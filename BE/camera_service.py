import logging
import time

import cv2

from pose_3d import bbox_to_pseudo_3d
from tracker import SimpleObjectTracker


LOGGER = logging.getLogger(__name__)


class CameraService:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.capture = None
        self.frame_id = 0
        self.tracker = SimpleObjectTracker()

    def open(self) -> bool:
        if self.capture and self.capture.isOpened():
            return True

        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened():
            LOGGER.warning("Cannot open camera index %s", self.camera_index)
            self.capture.release()
            self.capture = None
            return False

        return True

    def close(self) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None

    def read_frame(self):
        if not self.open():
            raise RuntimeError("Không mở được camera")

        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("Không đọc được frame từ camera")

        return frame

    def process_next_frame(self, detector, pose_estimator=None) -> dict:
        frame = self.read_frame()
        height, width = frame.shape[:2]
        timestamp = time.time()

        objects = []
        for obj in detector.detect(frame):
            obj["position_3d"] = bbox_to_pseudo_3d(
                obj["bbox_2d"],
                width,
                height,
                obj.get("confidence", 1.0),
            )
            objects.append(obj)

        poses = pose_estimator.estimate(frame) if pose_estimator else []
        for pose in poses:
            pose["position_3d"] = bbox_to_pseudo_3d(
                pose["bbox_2d"],
                width,
                height,
                pose.get("confidence", 1.0),
            )
            pose["source"] = "pose"
            objects.append(pose)

        objects = self.tracker.update(objects, self.frame_id, timestamp)
        response = {
            "success": True,
            "source_type": "camera",
            "frame_id": self.frame_id,
            "image_size": {"width": width, "height": height},
            "objects": objects,
        }
        self.frame_id += 1
        return response
