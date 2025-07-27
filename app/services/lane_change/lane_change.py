from __future__ import annotations
import os
import time
import math
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional
from fastapi.concurrency import run_in_threadpool

import easyocr

# ----------------------------------------
# Lane-related utilities

lane_lines = [
    # Define lane boundary lines here: (x1, y1), (x2, y2)
    ((30, 1120), (740, 520)),
    ((360, 1120), (820, 520)),
    ((740, 1120), (900, 520)),
]

def get_lane_bounds_at_y(y: int, lane_lines: List) -> List[float]:
    xs = []
    for (x1, y1), (x2, y2) in lane_lines:
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            x = x1 + (x2 - x1) * (y - y1) / (y2 - y1) if y2 != y1 else x1
            xs.append(x)
    xs.sort()
    return xs

def get_box_lane_overlap(x1: float, x2: float, lane_bounds: List[float]) -> List[float]:
    overlaps = []
    lane_width = x2 - x1
    for i in range(len(lane_bounds) - 1):
        left = max(x1, lane_bounds[i])
        right = min(x2, lane_bounds[i + 1])
        overlap = max(0, right - left)
        overlaps.append(overlap / lane_width if lane_width > 0 else 0)
    return overlaps

def compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (areaA + areaB - interArea + 1e-5)

# ----------------------------------------
# Vehicle tracking class

class VehicleTracker:
    def __init__(self, max_disappeared=140, min_distance=70):
        self.next_object_id = 0
        self.objects = {}  # object_id: box [x1,y1,x2,y2]
        self.disappeared = {}  # object_id: disappeared_count
        self.max_disappeared = max_disappeared
        self.min_distance = min_distance

    def register(self, box):
        self.objects[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, detections):
        updated = []

        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return updated

        input_centroids = np.zeros((len(detections), 2), dtype="float")
        input_boxes = []
        for i, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            input_centroids[i] = (cx, cy)
            input_boxes.append((x1, y1, x2, y2, conf, cls))

        if len(self.objects) == 0:
            for i in range(len(input_boxes)):
                oid = self.register(input_boxes[i][:4])
                updated.append((oid, input_boxes[i]))
        else:
            object_ids = list(self.objects.keys())
            iou_matrix = np.zeros((len(object_ids), len(input_boxes)), dtype=np.float32)

            for i, oid in enumerate(object_ids):
                prev_box = self.objects[oid]
                for j, det in enumerate(input_boxes):
                    curr_box = det[:4]
                    iou_matrix[i, j] = compute_iou(prev_box, curr_box)

            used_rows, used_cols = set(), set()
            for i in range(len(object_ids)):
                best_match = np.argmax(iou_matrix[i])
                max_iou = iou_matrix[i, best_match]
                if best_match in used_cols or max_iou < 0.3:
                    continue
                oid = object_ids[i]
                self.objects[oid] = input_boxes[best_match][:4]
                self.disappeared[oid] = 0
                updated.append((oid, input_boxes[best_match]))
                used_rows.add(i)
                used_cols.add(best_match)

            unused_rows = set(range(iou_matrix.shape[0])) - used_rows
            for row in unused_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            unused_cols = set(range(iou_matrix.shape[1])) - used_cols
            for col in unused_cols:
                oid = self.register(input_boxes[col][:4])
                updated.append((oid, input_boxes[col]))

        return updated

# ----------------------------------------
# Main U-turn detection service class

class UTurnDetectionService:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5, output_dir: str = "outputs"):
        from ultralytics import YOLO  # Must be installed separately
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.tracker = VehicleTracker(max_disappeared=140, min_distance=70)

        self.vehicle_tracks = defaultdict(list)  # id: list of normalized positions (x,y)
        self.u_turn_vehicles = set()
        self.frame_buffer = deque(maxlen=60)
        self.vehicle_frame_boxes = deque(maxlen=60)
        self.vehicle_lane_history = defaultdict(list)
        self.lane_changed_vehicles = set()
        self.plate_reader = easyocr.Reader(['en'])

        self.frame_shape: Optional[tuple[int, int]] = None
        self.min_track_length = 6

    def detect_vehicles(self, frame) -> List[List[float]]:
        results = self.model(frame, conf=self.confidence, classes=self.vehicle_classes)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                detections.append([x1, y1, x2, y2, conf, cls])
        return detections

    def update_tracks(self, tracking_results, frame_shape):
        height, width = frame_shape
        self.frame_shape = frame_shape

        # Append None for vehicles no longer detected (will replicate last known position below)
        for oid in self.vehicle_tracks:
            self.vehicle_tracks[oid].append(None)

        for oid, (x1, y1, x2, y2, conf, cls) in tracking_results:
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            self.vehicle_tracks[oid].append((cx, cy))

    def smooth_track(self, track, k=2):
        smoothed = []
        for i in range(len(track)):
            window = track[max(0, i-k):i+k+1]
            avg_x = sum(p[0] for p in window) / len(window)
            avg_y = sum(p[1] for p in window) / len(window)
            smoothed.append((avg_x, avg_y))
        return smoothed

    def detect_u_turns(self, vertical_threshold=0.15, horizontal_threshold=0.15) -> Dict[int, Dict[str, Any]]:
        u_turns = {}
        if not self.frame_shape:
            return u_turns
        height, width = self.frame_shape

        for track_id, track in self.vehicle_tracks.items():
            # Clean track removing None values, require minimum length
            cleaned_track = [pos for pos in track if pos is not None]
            if len(cleaned_track) < self.min_track_length:
                continue

            state = 0  # 0=initial vertical, 1=horizontal, 2=opposite vertical
            y_start = cleaned_track[0][1]
            vertical_direction = None
            v1_end_idx = None
            v2_start_idx = None
            v2_end_idx = None
            x_horiz_start = None
            y_middle = None
            y_horiz_end = None

            for i in range(1, len(cleaned_track)):
                x_curr, y_curr = cleaned_track[i]
                delta_x = x_curr - cleaned_track[i - 1][0]
                delta_y = y_curr - cleaned_track[i - 1][1]

                if state == 0:
                    if abs(y_curr - y_start) > vertical_threshold:
                        vertical_direction = np.sign(y_curr - y_start)
                        v1_end_idx = i
                        state = 1
                        x_horiz_start = x_curr
                        y_middle = y_curr
                elif state == 1:
                    if abs(x_curr - x_horiz_start) > horizontal_threshold:
                        state = 2
                        v2_start_idx = i
                        y_horiz_end = y_curr
                elif state == 2:
                    if abs(y_curr - y_horiz_end) > vertical_threshold:
                        new_vertical_direction = np.sign(y_curr - y_horiz_end)
                        v2_end_idx = i
                        if new_vertical_direction != vertical_direction:
                            # Calculate vectors of vertical segments and angle between
                            v1 = np.array([0, y_middle - y_start])
                            v2 = np.array([0, y_curr - y_middle])
                            dot = np.dot(v1, v2)
                            norm_v1 = np.linalg.norm(v1)
                            norm_v2 = np.linalg.norm(v2)
                            if norm_v1 > 0 and norm_v2 > 0:
                                angle = math.degrees(math.acos(dot / (norm_v1 * norm_v2)))
                                if angle > 150:  # Sharp angle -> U-turn
                                    u_turns[track_id] = {
                                        'point': cleaned_track[i],
                                        'frame_index': i,
                                        'position': (int(x_curr * width), int(y_curr * height)),
                                        'angle': angle,
                                    }
                                    self.u_turn_vehicles.add(track_id)
                                    break

        return u_turns

    def draw_tracks(self, frame, color=(0, 255, 0), thickness=2):
        h, w = frame.shape[:2]
        for track_id, points in self.vehicle_tracks.items():
            pts = [p for p in points if p is not None]
            if len(pts) < 2:
                continue
            for i in range(1, len(pts)):
                start = (int(pts[i-1][0] * w), int(pts[i-1][1] * h))
                end = (int(pts[i][0] * w), int(pts[i][1] * h))
                track_color = (0, 0, 255) if track_id in self.u_turn_vehicles else color
                cv2.line(frame, start, end, track_color, thickness)
        return frame

    def draw_u_turns(self, frame, u_turns):
        h, w = frame.shape[:2]
        for track_id, info in u_turns.items():
            x, y = info['position']
            cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)  # red filled circle
            cv2.circle(frame, (x, y), 20, (255, 255, 255), 2)  # white outline
            arrow_size = 15
            cv2.arrowedLine(frame, (x - arrow_size, y), (x + arrow_size, y),
                            (255, 255, 255), 2, tipLength=0.3)
            cv2.arrowedLine(frame, (x + arrow_size, y), (x - arrow_size, y),
                            (255, 255, 255), 2, tipLength=0.3)
            cv2.putText(frame, f"U-Turn: {info['angle']:.1f}°", (x - 60, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def draw_info(self, frame, frame_idx, frame_count, fps, proc_time):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (350, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_idx}/{frame_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {len(self.vehicle_tracks)}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"U-turns: {len(self.u_turn_vehicles)}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        fps_text = f"FPS: {1/proc_time:.1f}" if proc_time > 0 else "FPS: N/A"
        cv2.putText(frame, fps_text, (w - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def process_video_sync(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show_display: bool = False,
        save_frequency: int = 1,
    ) -> Dict[str, Any]:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.frame_shape = (height, width)

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps / save_frequency, (width, height))

        frame_idx = 0
        u_turn_events = []

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if frame_idx % 10 == 0 and total_frames > 0:
                    print(f"Processing frame {frame_idx}/{total_frames} ({100 * frame_idx / total_frames:.1f}%)")

                self.frame_buffer.append(frame.copy())
                detections = self.detect_vehicles(frame)
                tracking_results = self.tracker.update(detections)

                active_ids = [oid for oid, _ in tracking_results]

                # Fill missing positions for lost tracks with last known pos to keep path continuous
                for oid in self.tracker.objects.keys():
                    if oid not in active_ids:
                        if oid in self.vehicle_tracks and len(self.vehicle_tracks[oid]) > 0:
                            last_pos = self.vehicle_tracks[oid][-1]
                            self.vehicle_tracks[oid].append(last_pos)

                self.update_tracks(tracking_results, (height, width))

                # Update lane history and lane-change detection
                for oid, (x1, y1, x2, y2, conf, cls) in tracking_results:
                    cy = (y1 + y2) / 2
                    lane_bounds = get_lane_bounds_at_y(cy, lane_lines)
                    if len(lane_bounds) < 2:
                        continue
                    overlaps = get_box_lane_overlap(x1, x2, lane_bounds)
                    lane_idx = np.argmax(overlaps)
                    overlap_frac = overlaps[lane_idx]
                    if overlap_frac < 0.05:
                        lane_idx = -1
                    self.vehicle_lane_history[oid].append(lane_idx)

                    hist = self.vehicle_lane_history[oid]
                    if len(hist) >= 2:
                        if hist[-2] != -1 and hist[-1] != -1 and hist[-2] != hist[-1]:
                            self.lane_changed_vehicles.add(oid)

                # Draw bounding boxes colored by status
                for oid, (x1, y1, x2, y2, conf, cls) in tracking_results:
                    col = (0, 255, 0)  # green
                    if oid in self.lane_changed_vehicles:
                        col = (0, 0, 255)  # red
                    if oid in self.u_turn_vehicles:
                        col = (255, 0, 0)  # blue overrides red
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                    cv2.putText(frame, f"ID: {oid}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

                # Store boxes for OCR lookup on past frames
                current_boxes = {oid: (x1, y1, x2, y2) for oid, (x1, y1, x2, y2, _, _) in tracking_results}
                self.vehicle_frame_boxes.append(current_boxes)

                # Keep alive tracks for objects not detected this frame
                for oid in self.tracker.objects:
                    if oid not in active_ids:
                        x1, y1, x2, y2 = self.tracker.objects[oid]
                        cx = (x1 + x2) / 2 / width
                        cy = (y1 + y2) / 2 / height
                        self.vehicle_tracks[oid].append((cx, cy))

                # Detect U-turns
                u_turns = self.detect_u_turns()
                for tid, info in u_turns.items():
                    if not any(ev['track_id'] == tid for ev in u_turn_events):
                        timestamp = frame_idx / fps
                        position = info['position']
                        u_turn_events.append({
                            'track_id': tid,
                            'timestamp': timestamp,
                            'frame_idx': frame_idx,
                            'angle': info['angle'],
                            'position': position,
                        })

                        # OCR plate recognition lookback approx 1 sec ago (30 frames)
                        lookback = 30
                        if len(self.frame_buffer) > lookback and len(self.vehicle_frame_boxes) > lookback:
                            hist_frame = self.frame_buffer[-lookback]
                            hist_boxes = self.vehicle_frame_boxes[-lookback]
                            box = hist_boxes.get(tid)
                            if box is not None:
                                x1, y1, x2, y2 = box
                                vehicle_crop = hist_frame[int(y1):int(y2), int(x1):int(x2)]
                                plate_results = self.plate_reader.readtext(vehicle_crop)
                                if plate_results:
                                    best_plate = max(plate_results, key=lambda x: x[2])
                                    u_turn_events[-1]['plate'] = best_plate[1]  # plate text

                processing_time = time.time() - start_time

                frame = self.draw_tracks(frame)
                frame = self.draw_u_turns(frame, u_turns)
                frame = self.draw_info(frame, frame_idx, total_frames, fps, processing_time)

                if writer and frame_idx % save_frequency == 0:
                    writer.write(frame)

                if show_display:
                    max_h, max_w = 900, 1600
                    h, w = frame.shape[:2]
                    if h > max_h or w > max_w:
                        scale = min(max_h / h, max_w / w)
                        resized = cv2.resize(frame, None, fx=scale, fy=scale)
                        cv2.imshow("U-Turn Detection", resized)
                    else:
                        cv2.imshow("U-Turn Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        return {
            "video_path": video_path,
            "output_video": output_path,
            "total_frames": frame_idx,
            "u_turn_events": u_turn_events,
            "total_u_turns": len(u_turn_events),
        }

    async def process_video(self, video_path: str, output_path: Optional[str] = None, show_display: bool = False, save_frequency: int = 1):
        return await run_in_threadpool(self.process_video_sync, video_path, output_path, show_display, save_frequency)

# ----------------------------------------
# Convenience global service cache and helper

_service_cache: Optional[UTurnDetectionService] = None

async def run_u_turn_detection_async(video_path: str, config: Optional[Dict[str, Any]] = None):
    global _service_cache
    if _service_cache is None or config is not None:
        model_path = config.get("model_path") if config else "yolov8n.pt"
        confidence = config.get("confidence", 0.5) if config else 0.5
        output_dir = config.get("output_dir", "outputs") if config else "outputs"
        _service_cache = UTurnDetectionService(model_path=model_path, confidence=confidence, output_dir=output_dir)
    return await _service_cache.process_video(video_path, show_display=False)

