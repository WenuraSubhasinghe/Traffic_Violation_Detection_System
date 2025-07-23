from typing import List, Dict, Any
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections: List[Dict[str, Any]], frame) -> List[Dict[str, Any]]:
        # Convert to DeepSort's expected format
        det_list = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = d.get("confidence", 1.0)
            det_list.append([[x1, y1, w, h], conf])

        # Run DeepSort
        tracks = self.tracker.update_tracks(det_list if det_list else [], frame=frame)

        # Convert DeepSort tracks to simple dicts
        normalized_tracks: List[Dict[str, Any]] = []
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            l, t, r, b = tr.to_ltrb()
            normalized_tracks.append({
                "track_id": tr.track_id,
                "bbox": [int(l), int(t), int(r), int(b)]
            })
        return normalized_tracks