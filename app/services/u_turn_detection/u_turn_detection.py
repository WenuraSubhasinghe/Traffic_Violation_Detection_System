import cv2
import numpy as np
import os
import time
import easyocr
import re
from collections import defaultdict, deque

def normalize_plate(text):
    """Normalize detected plate text to standard format."""
    text = text.replace('=', '-').replace(' ', '').upper()
    if re.match(r'^\d+-\d+$', text):  # numbers-numbers
        return text
    if re.match(r'^[A-Z]{2}-\d+$', text):  # two letters-numbers
        return text
    return None

def compute_iou(boxA, boxB):
    """Compute Intersection-over-Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def smooth_track(track, k=2):
    """
    Smooth trajectory points with moving average filter.
    k controls smoothing window size (2k+1).
    """
    smoothed = []
    for i in range(len(track)):
        start = max(0, i - k)
        end = min(len(track), i + k + 1)
        window = track[start:end]
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        smoothed.append((avg_x, avg_y))
    return smoothed

class VehicleTracker:
    """Simple IoU-based vehicle tracker."""

    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = dict()  # object_id: bounding box
        self.disappeared = dict()
        self.max_disappeared = max_disappeared

    def register(self, box):
        self.objects[self.next_object_id] = box
        self.disappeared[self.next_object_id] = 0
        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Args:
            detections: list of boxes [x1, y1, x2, y2, conf, cls]

        Returns:
            List of tuples: (object_id, detection_box)
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return []

        input_boxes = [det[:4] for det in detections]
        input_boxes_conf_cls = detections

        if len(self.objects) == 0:
            results = []
            for i in range(len(input_boxes)):
                object_id = self.register(input_boxes[i])
                results.append((object_id, detections[i]))
            return results

        object_ids = list(self.objects.keys())
        object_boxes = list(self.objects.values())

        iou_matrix = np.zeros((len(object_boxes), len(input_boxes)), dtype=np.float32)

        for i, prev_box in enumerate(object_boxes):
            for j, curr_box in enumerate(input_boxes):
                iou_matrix[i, j] = compute_iou(prev_box, curr_box)

        used_rows, used_cols = set(), set()
        results = []

        for i in range(len(object_boxes)):
            best_match = np.argmax(iou_matrix[i])
            max_iou = iou_matrix[i, best_match]
            if best_match in used_cols or max_iou < 0.3:
                continue
            object_id = object_ids[i]
            self.objects[object_id] = input_boxes[best_match]
            self.disappeared[object_id] = 0
            results.append((object_id, input_boxes_conf_cls[best_match]))
            used_rows.add(i)
            used_cols.add(best_match)

        unused_rows = set(range(iou_matrix.shape[0])) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(iou_matrix.shape[1])) - used_cols
        for col in unused_cols:
            object_id = self.register(input_boxes[col])
            results.append((object_id, input_boxes_conf_cls[col]))

        return results

class UTurnDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Please install ultralytics (`pip install ultralytics`) to use this detector")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = [2,3,5,7]  # car, motorcycle, bus, truck (COCO classes)

        self.tracker = VehicleTracker(max_disappeared=140)

        self.vehicle_tracks = defaultdict(list)  # object_id -> list of norm (x,y)
        self.u_turn_vehicles = set()
        self.frame_buffer = deque(maxlen=60)
        self.vehicle_frame_boxes = deque(maxlen=60)  # frame idx -> {object_id: box}
        self.plate_reader = easyocr.Reader(['en'], gpu=False)
        self.min_track_length = 6
        self.frame_shape = None  # (height, width)
        self.vehicle_plates = dict()

    def detect_vehicles(self, frame):
        try:
            results = self.model(frame, conf=self.confidence, classes=self.vehicle_classes)
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = box.cls[0].item()
                    detections.append([x1, y1, x2, y2, conf, cls])
            return detections
        except Exception as e:
            print(f"[ERROR] Vehicle detection failed: {e}")
            return []

    def update_tracks(self, tracking_results, frame_shape):
        height, width = frame_shape
        self.frame_shape = frame_shape
        # Mark missing boxes
        for object_id in self.vehicle_tracks:
            self.vehicle_boxes.setdefault(object_id, []).append(None)
        # Update with detected
        for object_id, (x1, y1, x2, y2, conf, cls) in tracking_results:
            cx = (x1 + x2)/2.0
            cy = (y1 + y2)/2.0
            norm_x = cx / width
            norm_y = cy / height
            self.vehicle_tracks[object_id].append((norm_x, norm_y))
            self.vehicle_boxes.setdefault(object_id, []).append((x1, y1, x2, y2))

    def detect_u_turns(self, vertical_threshold=0.02, horizontal_threshold=0.02):
        u_turns = {}
        if not self.frame_shape:
            return u_turns
        height, width = self.frame_shape

        for track_id, track in self.vehicle_tracks.items():
            if len(track) < self.min_track_length:
                continue
            # Smooth track points to reduce noise
            track = smooth_track(track, k=2)

            state = 0
            y_start = track[0][1]
            vertical_direction = None
            for i in range(1, len(track)):
                x_curr, y_curr = track[i]
                if state == 0:
                    if abs(y_curr - y_start) > vertical_threshold:
                        vertical_direction = np.sign(y_curr - y_start)
                        state = 1
                        x_start = x_curr
                elif state == 1:
                    if abs(x_curr - x_start) > horizontal_threshold:
                        state = 2
                        y_middle = y_curr
                elif state == 2:
                    if abs(y_curr - y_middle) > vertical_threshold:
                        new_vertical_direction = np.sign(y_curr - y_middle)
                        if new_vertical_direction != vertical_direction:
                            # U-turn Detected
                            u_turns[track_id] = {
                                'point': track[i],
                                'frame_index': i,
                                'position': (int(x_curr * width), int(y_curr * height)),
                                'angle': 180
                            }
                            self.u_turn_vehicles.add(track_id)
                            break
        return u_turns

    def draw_tracks(self, frame, color=(0, 255, 0), thickness=2):
        h, w = frame.shape[:2]
        for track_id, points in self.vehicle_tracks.items():
            if len(points) < 2:
                continue
            for i in range(1, len(points)):
                start_pt = (int(points[i-1][0]*w), int(points[i-1][1]*h))
                end_pt = (int(points[i][0]*w), int(points[i][1]*h))
                track_color = (0,0,255) if track_id in self.u_turn_vehicles else color
                cv2.line(frame, start_pt, end_pt, track_color, thickness)
        return frame

    def draw_u_turns(self, frame, u_turns):
        h, w = frame.shape[:2]
        for track_id, uturn in u_turns.items():
            x = int(uturn['point'][0]*w)
            y = int(uturn['point'][1]*h)
            cv2.circle(frame, (x,y), 20, (0,0,255), -1)
            cv2.circle(frame, (x,y), 20, (255,255,255), 2)
            arrow_len = 15
            cv2.arrowedLine(frame, (x - arrow_len, y), (x + arrow_len, y), (255,255,255), 2, tipLength=0.3)
            cv2.arrowedLine(frame, (x + arrow_len, y), (x - arrow_len, y), (255,255,255), 2, tipLength=0.3)
            cv2.putText(frame, f"U-Turn: {uturn['angle']:.1f}°", (x - 60, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return frame

    def draw_info(self, frame, frame_idx, frame_count, fps, processing_time):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (350, 120), (0,0,0), -1)
        cv2.putText(frame, f"Frame: {frame_idx}/{frame_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        cv2.putText(frame, f"Vehicles: {len(self.vehicle_tracks)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        cv2.putText(frame, f"U-turns: {len(self.u_turn_vehicles)}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        fps_text = f"FPS: {1/processing_time:.1f}" if processing_time > 0 else "FPS: N/A"
        cv2.putText(frame, fps_text, (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
        return frame

    def process_video(self, video_path, output_path=None, show_display=False, save_frequency=1):
        self.vehicle_tracks.clear()
        self.u_turn_vehicles.clear()
        self.vehicle_plates.clear()
        self.vehicle_boxes = defaultdict(list)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_shape = (height, width)

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps / save_frequency, (width, height))

        u_turn_events = []
        frame_idx = 0

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                self.frame_buffer.append(frame.copy())

                detections = self.detect_vehicles(frame)
                tracking_results = self.tracker.update(detections)

                # Fill missing tracked positions with last known
                active_ids = {obj_id for obj_id, _ in tracking_results}
                for obj_id in self.tracker.objects.keys():
                    if obj_id not in active_ids:
                        if obj_id in self.vehicle_tracks and len(self.vehicle_tracks[obj_id]) > 0:
                            last_pos = self.vehicle_tracks[obj_id][-1]
                            self.vehicle_tracks[obj_id].append(last_pos)

                self.update_tracks(tracking_results, (height, width))

                current_boxes = {obj_id: box[:4] for obj_id, box in tracking_results}
                self.vehicle_frame_boxes.append(current_boxes)

                # Keep last known positions for tracked but non-detected vehicles
                for obj_id in self.tracker.objects:
                    if obj_id in active_ids:
                        continue
                    x1, y1, x2, y2 = self.tracker.objects[obj_id]
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    norm_x = cx / width
                    norm_y = cy / height
                    self.vehicle_tracks[obj_id].append((norm_x, norm_y))

                u_turns = self.detect_u_turns()

                for track_id, uturn_info in u_turns.items():
                    if not any(event['track_id'] == track_id for event in u_turn_events):
                        timestamp = frame_idx / fps
                        u_turn_events.append({
                            'track_id': track_id,
                            'timestamp': timestamp,
                            'frame_idx': frame_idx,
                            'angle': uturn_info['angle'],
                            'position': uturn_info['position'],
                        })

                        # Plate recognition from 30 frames ago
                        lookback = 30
                        if len(self.frame_buffer) > lookback and len(self.vehicle_frame_boxes) > lookback:
                            hist_frame = self.frame_buffer[-lookback]
                            hist_boxes = self.vehicle_frame_boxes[-lookback]
                            box = hist_boxes.get(track_id, None)
                            if box:
                                x1, y1, x2, y2 = map(int, box)
                                vehicle_crop = hist_frame[y1:y2, x1:x2]
                                if vehicle_crop.size == 0:
                                    continue
                                os.makedirs("saved_vehicles", exist_ok=True)
                                filename = f"vehicle_{track_id}_frame{frame_idx}_time{int(timestamp)}.jpg"
                                save_path = os.path.join("saved_vehicles", filename)
                                cv2.imwrite(save_path, vehicle_crop)
                                ocr_results = self.plate_reader.readtext(vehicle_crop)
                                normalized_plate = None
                                if ocr_results:
                                    best_result = max(ocr_results, key=lambda x: x[2])  # highest confidence
                                    normalized_plate = normalize_plate(best_result[1])
                                    if normalized_plate and track_id not in self.vehicle_plates:
                                        self.vehicle_plates[track_id] = normalized_plate
                                u_turn_events[-1]['plate'] = self.vehicle_plates.get(track_id, 'N/A')
                            else:
                                u_turn_events[-1]['plate'] = 'N/A'
                        else:
                            u_turn_events[-1]['plate'] = 'N/A'

                # Draw boxes and labels
                for obj_id, (x1, y1, x2, y2, conf, cls) in tracking_results:
                    color = (0,0,255) if obj_id in self.u_turn_vehicles else (0,255,0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"ID: {obj_id}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                processing_time = time.time() - start_time
                frame = self.draw_tracks(frame)
                frame = self.draw_u_turns(frame, u_turns)
                frame = self.draw_info(frame, frame_idx, frame_count, fps, processing_time)

                if writer and frame_idx % save_frequency == 0:
                    writer.write(frame)

                if show_display:
                    cv2.imshow("U-turn Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            print(f"[ERROR] Video processing failed: {e}")
            import traceback; traceback.print_exc()
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        # Summary output
        print(f"\nU-Turn detection complete: {len(u_turn_events)} events detected.")
        for i, ev in enumerate(u_turn_events):
            mins = int(ev['timestamp'] // 60)
            secs = int(ev['timestamp'] % 60)
            plate = ev.get('plate', 'N/A')
            print(f"{i+1}. Vehicle ID {ev['track_id']} made a {ev['angle']:.1f}° U-turn at {mins}:{secs:02d} Plate: {plate}")

        return u_turn_events
