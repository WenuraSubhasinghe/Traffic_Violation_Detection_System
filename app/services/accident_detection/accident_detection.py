from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import shutil
from fastapi import UploadFile

import cv2
import numpy as np

from fastapi.concurrency import run_in_threadpool

from app.database import db
from app.services.accident_detection.accident_config import AccidentConfig
from app.services.accident_detection.yolo_detector import YoloAccidentDetector, RED, YELLOW, GREEN
from app.services.accident_detection.tracker import VehicleTracker
from app.services.accident_detection.collision_aabb import CollisionDetector
from app.utils.video_converter import convert_to_browser_compatible


class AccidentDetectionService:
    """High-level orchestrator for accident detection on video files."""

    def __init__(self, config: Optional[AccidentConfig] = None, output_dir: str = "outputs"):
        self.config = config or AccidentConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.config.accident_frames_dir, exist_ok=True)

        self.detector = YoloAccidentDetector(
            vehicle_model_path=self.config.vehicle_model_path,
            accident_model_path=self.config.accident_model_path,
            vehicle_classes=self.config.vehicle_classes,
            vehicle_conf=self.config.vehicle_confidence_threshold,
        )
        self.tracker = VehicleTracker(
            max_age=self.config.tracker_max_age,
            n_init=self.config.tracker_n_init,
        )
        self.collision_detector = CollisionDetector(self.config.collision_threshold_percent)

    # ------------------------------------------------------------------
    # PUBLIC API -------------------------------------------------------
    # ------------------------------------------------------------------
    def process_video_sync(self, video_path: str) -> Dict[str, Any]:
        """Synchronous heavy lifting; safe to call in a thread."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        out_path = None
        writer = None
        if self.config.save_annotated_video:
            out_path = os.path.join(self.output_dir, f"annotated_{Path(video_path).stem}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        accident_frames = []  # list of {frame, timestamp, confidence, track_ids, bbox, path}
        collision_log = []     # log every collision check that triggered model

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Progress print (optional)
            if self.config.display_progress_interval and total_frames and frame_idx % self.config.display_progress_interval == 0:
                pct = 100.0 * frame_idx / total_frames
                print(f"Processing frame {frame_idx}/{total_frames} ({pct:.1f}%)")

            # 1. Detect vehicles
            detections = self.detector.detect(frame)

            # 2. Track
            tracks = self.tracker.update(detections, frame)

            # 3. Collision detect
            collisions = self.collision_detector.check(tracks)

            accident_results = []
            for col in collisions:
                # 4. Confirm accident
                is_acc, conf, acc_bbox = self.detector.confirm_accident(
                    frame,
                    col["bbox"],
                    confidence_threshold=self.config.accident_confidence_threshold,
                    padding=self.config.collision_padding,
                )

                accident_results.append({
                    "is_accident": is_acc,
                    "confidence": conf,
                    "bbox": acc_bbox,
                    "collision_region": col,
                })

                collision_log.append({
                    "frame": frame_idx,
                    "timestamp": frame_idx / fps,
                    "collision_region": col,
                    "is_accident": is_acc,
                    "confidence": conf,
                })

                # Save frame if accident confirmed
                if is_acc and self.config.save_accident_frames:
                    img_name = f"accident_frame_{frame_idx}_conf_{conf:.2f}.jpg"
                    img_path = os.path.join(self.config.accident_frames_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    accident_frames.append({
                        "frame": frame_idx,
                        "timestamp": frame_idx / fps,
                        "confidence": conf,
                        "track_ids": list(col["track_ids"]),
                        "bbox": acc_bbox,
                        "path": img_path,
                    })

            # 5. Annotate frame
            annotated = self._annotate(frame, tracks, collisions, accident_results)
            if writer is not None:
                writer.write(annotated)

        cap.release()
        if writer is not None:
            writer.release()
            out_path = convert_to_browser_compatible(out_path, overwrite=True)

        results = {
            "video_path": video_path,
            "output_video": out_path,
            "total_frames": frame_idx,
            "video_duration": (frame_idx / fps) if fps else None,
            "accident_frames": accident_frames,
            "collision_log": collision_log,
            "total_accidents": len([a for a in accident_frames if a]),
            "total_collisions": len(collision_log),
        }
        return results

    async def process_video(self, video_path: str) -> Dict[str, Any]:
        """Async wrapper; runs heavy sync code in a thread, then persists to DB."""
        results = await run_in_threadpool(self.process_video_sync, video_path)

        # Persist to MongoDB
        doc = {
            **results,
            "created_at": datetime.utcnow(),
        }
        insert_res = await db.accident_logs.insert_one(doc)
        doc_id = insert_res.inserted_id
        doc["_id"] = doc_id
        return doc
    
    async def test_single_image(self, image_path: str):
        """
        Run accident detection on a single image.
        Returns detection results and annotated image path.
        """
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Image could not be loaded.")

        # Step 1: Detect vehicles
        detections = self.detector.detect(frame)

        # Step 2: Track (single frame, just treat detections as tracks)
        tracks = [{"track_id": idx + 1, "bbox": d["bbox"]} for idx, d in enumerate(detections)]

        # Step 3: Collision detection
        collisions = self.collision_detector.check(tracks)

        # Step 4: Accident confirmation
        accident_results = []
        for col in collisions:
            is_acc, conf, acc_bbox = self.detector.confirm_accident(
                frame,
                col["bbox"],
                confidence_threshold=self.config.accident_confidence_threshold,
                padding=self.config.collision_padding,
            )
            accident_results.append({
                "is_accident": is_acc,
                "confidence": conf,
                "bbox": acc_bbox,
                "collision_region": col,
            })

        # Step 5: Annotate frame
        annotated = self._annotate(frame, tracks, collisions, accident_results)
        annotated_path = os.path.join(self.output_dir, f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(annotated_path, annotated)

        return {
            "detections": detections,
            "collisions": collisions,
            "accidents": accident_results,
            "annotated_image": annotated_path,
        }


    # ------------------------------------------------------------------
    # INTERNAL: annotation (draw vehicles, collisions, accidents)
    # ------------------------------------------------------------------
    def _annotate(self, frame, tracks, collisions, accident_results):
        annotated = frame.copy()

        # Vehicles (green)
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(annotated, f"ID:{t['track_id']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

        # Collision regions (yellow)
        for col in collisions:
            x1, y1, x2, y2 = col["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW, 2)
            cv2.putText(annotated, "COLLISION?", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

        # Accidents (red)
        for res in accident_results:
            if res["is_accident"] and res["bbox"] is not None:
                x1, y1, x2, y2 = res["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), RED, 3)
                cv2.putText(annotated, f"ACC! {res['confidence']:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        return annotated


# ----------------------------------------------------------------------
# Convenience function for one-off async calls without manual service mgmt
# ----------------------------------------------------------------------
_service_cache: Optional[AccidentDetectionService] = None

async def run_accident_detection_async(video_path: str, config: Optional[AccidentConfig] = None):
    global _service_cache
    if _service_cache is None or config is not None:
        _service_cache = AccidentDetectionService(config=config)
    return await _service_cache.process_video(video_path)