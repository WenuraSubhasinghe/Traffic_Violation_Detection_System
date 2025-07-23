from __future__ import annotations
from typing import Dict, List, Tuple, Any

class CollisionDetector:
    """Axis-Aligned Bounding Box (AABB) collision & near-collision checks.

    Works on *tracked objects* that expose a `bbox` in [x1, y1, x2, y2] pixel coords.
    You *may* pass raw DeepSORT Track objects; we provide an adapter.
    """

    def __init__(self, threshold_percent: float = 0.05):
        self.threshold_percent = threshold_percent

    @staticmethod
    def _calc_centers(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def _calc_half_dims(box):
        x1, y1, x2, y2 = box
        return (x2 - x1) / 2, (y2 - y1) / 2

    def _aabb_collision(self, box1, box2) -> Tuple[bool, Dict[str, float]]:
        cx1, cy1 = self._calc_centers(box1)
        cx2, cy2 = self._calc_centers(box2)
        hw1, hh1 = self._calc_half_dims(box1)
        hw2, hh2 = self._calc_half_dims(box2)

        horiz_dist = abs(cx1 - cx2)
        vert_dist = abs(cy1 - cy2)
        horiz_overlap = horiz_dist - (hw1 + hw2)
        vert_overlap = vert_dist - (hh1 + hh2)

        width_thresh = min(hw1, hw2) * self.threshold_percent
        height_thresh = min(hh1, hh2) * self.threshold_percent

        collision = (horiz_overlap <= width_thresh) and (vert_overlap <= height_thresh)

        info = {
            "horizontal_overlap": horiz_overlap,
            "vertical_overlap": vert_overlap,
            "horizontal_distance": horiz_dist,
            "vertical_distance": vert_dist,
            "width_threshold": width_thresh,
            "height_threshold": height_thresh,
        }
        return collision, info

    def check(self, tracks: List[Dict[str, Any]]):
        """Check all track pairs for collisions.

        Parameters
        ----------
        tracks: list of dicts with keys:
            - track_id: int/str
            - bbox: [x1, y1, x2, y2]

        Returns
        -------
        collisions: list of dicts {bbox, track_ids:(id1,id2), distance_info:{...}}
        """
        collisions = []
        n = len(tracks)
        for i in range(n):
            for j in range(i + 1, n):
                t1 = tracks[i]
                t2 = tracks[j]
                box1 = t1["bbox"]
                box2 = t2["bbox"]
                collides, info = self._aabb_collision(box1, box2)
                if collides:
                    x1 = min(box1[0], box2[0])
                    y1 = min(box1[1], box2[1])
                    x2 = max(box1[2], box2[2])
                    y2 = max(box1[3], box2[3])
                    collisions.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "track_ids": (t1["track_id"], t2["track_id"]),
                        "distance_info": info,
                    })
        return collisions