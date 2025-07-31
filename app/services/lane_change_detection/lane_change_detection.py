import cv2
import numpy as np
import os
import easyocr
from collections import defaultdict
import time

try:
    from ultralytics import YOLO
except ImportError:
    print('Please install ultralytics (pip install ultralytics).')
    exit(1)

# --- Define your lane lines and colors here ---
lane_lines = [
    ((30, 1120), (740, 520)),
    ((360, 1120), (820, 520)),
    ((740, 1120), (900, 520)),
]
lane_colors = [
    (255, 200, 200),
    (200, 255, 200),
    (200, 200, 255),
    (255, 255, 200)
]

def get_lane_bounds_at_y(y, lane_lines):
    xs = []
    for (x1, y1), (x2, y2) in lane_lines:
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if y2 != y1:
                x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
            else:
                x = x1
            xs.append(x)
    xs.sort()
    return xs

def get_box_lane_overlap(x1, x2, lane_bounds):
    overlaps = []
    for i in range(len(lane_bounds) - 1):
        left = max(x1, lane_bounds[i])
        right = min(x2, lane_bounds[i+1])
        overlap = max(0, right - left)
        lane_width = x2 - x1
        overlaps.append(overlap / lane_width if lane_width > 0 else 0)
    return overlaps

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

class VehicleTracker:
    def __init__(self, max_disappeared=50, min_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.min_distance = min_distance

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
        updated_objects = []

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return updated_objects

        input_boxes = [det[:4] + det[4:] for det in detections]

        if len(self.objects) == 0:
            for det in input_boxes:
                object_id = self.register(det[:4])
                updated_objects.append((object_id, det))
            return updated_objects

        object_ids = list(self.objects.keys())
        object_boxes = list(self.objects.values())

        iou_matrix = np.zeros((len(object_ids), len(input_boxes)), dtype=np.float32)

        for i, obj_box in enumerate(object_boxes):
            for j, det_box in enumerate(input_boxes):
                iou_matrix[i, j] = compute_iou(obj_box, det_box[:4])

        used_rows, used_cols = set(), set()

        for i in range(len(object_ids)):
            best_match = np.argmax(iou_matrix[i])
            max_iou = iou_matrix[i, best_match]
            if best_match in used_cols or max_iou < 0.3:
                continue
            object_id = object_ids[i]
            self.objects[object_id] = input_boxes[best_match][:4]
            self.disappeared[object_id] = 0
            updated_objects.append((object_id, input_boxes[best_match]))
            used_rows.add(i)
            used_cols.add(best_match)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(len(input_boxes))) - used_cols
        for col in unused_cols:
            object_id = self.register(input_boxes[col][:4])
            updated_objects.append((object_id, input_boxes[col]))

        return updated_objects

class LaneChangeDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO)
        self.tracker = VehicleTracker(max_disappeared=50)
        self.vehicle_tracks = defaultdict(list)
        self.vehicle_boxes = defaultdict(list)
        self.vehicle_lane_history = defaultdict(list)
        self.lane_changed_vehicles = set()
        self.frame_shape = None

        # Plate reader for OCR
        self.plate_reader = easyocr.Reader(['en'], gpu=False)
        # Current frame for OCR cropping
        self.current_frame = None

    def detect_vehicles(self, frame):
        results = self.model(frame, conf=self.confidence, classes=self.vehicle_classes)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                detections.append([x1, y1, x2, y2, conf, cls])
        return detections

    def update_tracks(self, tracking_results, frame_shape):
        height, width = frame_shape
        self.frame_shape = frame_shape
        for object_id in self.vehicle_tracks:  # Add placeholder if not seen this frame
            self.vehicle_boxes[object_id].append(None)
        for object_id, (x1, y1, x2, y2, conf, cls) in tracking_results:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            norm_x, norm_y = cx / width, cy / height
            self.vehicle_tracks[object_id].append((norm_x, norm_y))
            self.vehicle_boxes[object_id].append((x1, y1, x2, y2))

    def assign_lanes_and_detect_changes(self, tracking_results, frame_idx, lane_change_events, fps):
        for object_id, (x1, y1, x2, y2, conf, cls) in tracking_results:
            cy = (y1 + y2) / 2
            lane_bounds = get_lane_bounds_at_y(cy, lane_lines)
            if len(lane_bounds) < 2:
                lane_idx = -1
            else:
                overlaps = get_box_lane_overlap(x1, x2, lane_bounds)
                lane_idx = int(np.argmax(overlaps))
                if overlaps[lane_idx] < 0.05:
                    lane_idx = -1
            self.vehicle_lane_history[object_id].append(lane_idx)
            lane_hist = self.vehicle_lane_history[object_id]
            if len(lane_hist) >= 2:
                prev_lane = lane_hist[-2]
                curr_lane = lane_hist[-1]
                if prev_lane != -1 and curr_lane != -1 and prev_lane != curr_lane:
                    if object_id not in self.lane_changed_vehicles:
                        self.lane_changed_vehicles.add(object_id)
                        timestamp = frame_idx / fps

                        # Plate recognition on current frame crop
                        plate_number = "N/A"
                        x1_int, y1_int, x2_int, y2_int = map(int, [x1, y1, x2, y2])
                        vehicle_crop = self.current_frame[y1_int:y2_int, x1_int:x2_int]
                        if vehicle_crop.size > 0:
                            try:
                                ocr_results = self.plate_reader.readtext(vehicle_crop)
                                if ocr_results:
                                    best = max(ocr_results, key=lambda x: x[2])  # confidence score max
                                    plate_candidate = best[1].replace('=', '-').replace(' ', '').upper()
                                    if len(plate_candidate) > 2:
                                        plate_number = plate_candidate
                            except Exception:
                                pass  # fail silently on OCR error

                        lane_change_events.append({
                            'track_id': object_id,
                            'timestamp': timestamp,
                            'frame_idx': frame_idx,
                            'from_lane': prev_lane,
                            'to_lane': curr_lane,
                            'plate_number': plate_number
                        })

    def draw_lanes_and_regions(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for pt1, pt2 in lane_lines:
            cv2.line(overlay, pt1, pt2, (255, 120, 0), 5)
        cv2.addWeighted(overlay, 0.05, frame, 0.95, 0, frame)
        lane_overlay = np.zeros_like(frame, dtype=np.uint8)
        for y in range(h):
            lane_bounds = get_lane_bounds_at_y(y, lane_lines)
            for i in range(len(lane_bounds) - 1):
                x_left = int(lane_bounds[i])
                x_right = int(lane_bounds[i + 1])
                cv2.line(lane_overlay, (x_left, y), (x_right, y), lane_colors[i % len(lane_colors)], 2)
        cv2.addWeighted(lane_overlay, 0.3, frame, 0.7, 0, frame)
        return frame

    def draw_tracks(self, frame):
        h, w = frame.shape[:2]
        for track_id, points in self.vehicle_tracks.items():
            if len(points) < 2:
                continue
            color = (0, 255, 0)
            if track_id in self.lane_changed_vehicles:
                color = (0, 0, 255)
            for i in range(1, len(points)):
                prev_point = (int(points[i-1][0] * w), int(points[i-1][1] * h))
                curr_point = (int(points[i][0] * w), int(points[i][1] * h))
                cv2.line(frame, prev_point, curr_point, color, 2)
        return frame

    def process_video(self, video_path, output_path=None, show_display=False, save_frequency=1):
        self.vehicle_tracks.clear()
        self.lane_changed_vehicles.clear()
        self.vehicle_lane_history.clear()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
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

        frame_idx = 0
        lane_change_events = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                self.current_frame = frame.copy()  # For OCR

                frame = self.draw_lanes_and_regions(frame)
                detections = self.detect_vehicles(frame)
                tracking_results = self.tracker.update(detections)
                self.update_tracks(tracking_results, (height, width))
                self.assign_lanes_and_detect_changes(tracking_results, frame_idx, lane_change_events, fps)
                frame = self.draw_tracks(frame)

                for object_id, det in tracking_results:
                    x1, y1, x2, y2, conf, cls = det
                    color = (0, 255, 0)
                    if object_id in self.lane_changed_vehicles:
                        color = (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID: {object_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.rectangle(frame, (10, 10), (370, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"Frame: {frame_idx}/{frame_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Vehicles: {len(self.vehicle_tracks)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Lane changes: {len(self.lane_changed_vehicles)}", (165, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                if writer and (frame_idx % save_frequency == 0):
                    writer.write(frame)

                if show_display:
                    max_display_height = 900
                    max_display_width = 1600
                    current_height, current_width = frame.shape[:2]
                    if current_height > max_display_height or current_width > max_display_width:
                        scale = min(max_display_height / current_height, max_display_width / current_width)
                        display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                        cv2.imshow('Lane Change Detection', display_frame)
                    else:
                        cv2.imshow('Lane Change Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            print(f"Error processing video: {e}", flush=True)
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        print(f"\nTotal lane changes detected: {len(lane_change_events)}")
        for i, event in enumerate(lane_change_events):
            plate_desc = f" (Plate: {event.get('plate_number', '')})" if event.get('plate_number') and event.get('plate_number') != "N/A" else ""
            mins = int(event['timestamp'] // 60)
            secs = int(event['timestamp'] % 60)
            print(f"{i + 1}. Vehicle ID {event['track_id']} changed from lane {event['from_lane']} to lane {event['to_lane']} at {mins}:{secs:02d}{plate_desc}")

        return lane_change_events
