from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# Colors (BGR) for annotation
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# COCO class ids -> names (subset)
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class YoloAccidentDetector:
    """Wrapper around YOLOv8 models for vehicle detection + accident confirmation."""

    def __init__(
        self,
        vehicle_model_path: str = "models/yolov8s.pt",
        accident_model_path: str = "models/accident_detection_model.pt",
        vehicle_classes = (2, 3, 5, 7),
        vehicle_conf=0.5,
    ):
        # Resolve & fallback
        veh_path = Path(vehicle_model_path)
        if not veh_path.exists():
            veh_path = Path("yolov8s.pt")  # fallback to default in cache if available
        self.vehicle_model = YOLO(str(veh_path))

        acc_path = Path(accident_model_path)
        if acc_path.exists():
            self.accident_model = YOLO(str(acc_path))
            self._accident_loaded = True
        else:
            self.accident_model = self.vehicle_model  # fallback
            self._accident_loaded = False

        self.vehicle_classes = set(vehicle_classes)
        self.vehicle_conf = vehicle_conf

    # ------------------------------------------------------------------
    # VEHICLE DETECTION
    # ------------------------------------------------------------------
    def detect(self, frame) -> List[Dict[str, Any]]:
        """Detect vehicles in a BGR frame; returns list of detection dicts."""
        results = self.vehicle_model(frame, conf=self.vehicle_conf, verbose=False)
        out = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                cls = int(b.cls[0])
                if cls not in self.vehicle_classes:
                    continue
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                out.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": CLASS_NAMES.get(cls, str(cls)),
                })
        return out

    # ------------------------------------------------------------------
    # ACCIDENT CONFIRMATION ON CROPPED REGION
    # ------------------------------------------------------------------
    def confirm_accident(
        self,
        frame,
        collision_bbox: List[int],
        confidence_threshold: float = 0.7,
        padding: int = 20,
    ) -> Tuple[bool, float, Optional[List[int]]]:
        """Run accident model on a crop around collision bbox.

        Returns (is_accident, confidence, full_frame_bbox or None)
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = collision_bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w - 1, x2 + padding)
        y2 = min(h - 1, y2 + padding)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0.0, None

        results = self.accident_model(crop, conf=confidence_threshold, verbose=False)
        best_conf = 0.0
        best_abs_bbox = None

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                conf = float(b.conf[0].cpu().numpy())
                if conf > best_conf:
                    rel_x1, rel_y1, rel_x2, rel_y2 = b.xyxy[0].cpu().numpy()
                    best_conf = conf
                    best_abs_bbox = [
                        int(x1 + rel_x1),
                        int(y1 + rel_y1),
                        int(x1 + rel_x2),
                        int(y1 + rel_y2),
                    ]

        is_accident = best_conf >= confidence_threshold
        return is_accident, best_conf, best_abs_bbox