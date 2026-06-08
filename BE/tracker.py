import math


def center_of_bbox(bbox_2d: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_2d
    return (x1 + x2) / 2, (y1 + y2) / 2


class SimpleObjectTracker:
    def __init__(self, max_distance: float = 80.0, max_missed_frames: int = 20):
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def update(self, objects: list[dict], frame_id: int, timestamp: float | None = None) -> list[dict]:
        assigned_tracks = set()

        for obj in objects:
            center = center_of_bbox(obj["bbox_2d"])
            best_track_id = None
            best_distance = self.max_distance

            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                if track["class_name"] != obj["class_name"]:
                    continue

                distance = math.dist(center, track["center"])
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_id
                self.next_id += 1
                velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
            else:
                previous = self.tracks[best_track_id]
                previous_position = previous.get("position_3d", {})
                current_position = obj.get("position_3d", {})
                delta = max(frame_id - previous.get("frame_id", frame_id), 1)
                velocity = {
                    axis: round(
                        (float(current_position.get(axis, 0.0)) - float(previous_position.get(axis, 0.0))) / delta,
                        4,
                    )
                    for axis in ("x", "y", "z")
                }

            assigned_tracks.add(best_track_id)
            obj["object_id"] = best_track_id
            obj["velocity"] = velocity
            obj["frame_id"] = frame_id
            if timestamp is not None:
                obj["timestamp"] = round(timestamp, 4)

            self.tracks[best_track_id] = {
                "class_name": obj["class_name"],
                "center": center,
                "position_3d": obj.get("position_3d", {}),
                "frame_id": frame_id,
                "missed": 0,
            }

        for track_id in list(self.tracks):
            if track_id not in assigned_tracks:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missed_frames:
                    del self.tracks[track_id]

        return objects
