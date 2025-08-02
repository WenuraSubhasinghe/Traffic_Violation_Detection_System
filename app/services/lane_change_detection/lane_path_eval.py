import cv2
import numpy as np
import os
from collections import defaultdict
try:
    from ultralytics import YOLO
except ImportError:
    print('Please install ultralytics (pip install ultralytics).')
    exit(1)

class LaneChangeDetector:
    def __init__(self, lane_model_path, vehicle_model_path, confidence=0.5):
        self.lane_model = YOLO(lane_model_path)
        self.vehicle_model = YOLO(vehicle_model_path)
        self.confidence = confidence
        self.tracker = self.VehicleTracker(max_disappeared=50)
        self.vehicle_classes = [2, 3, 5, 7]  # COCO vehicle IDs
        self.lane_polygons = None
        self.vehicle_tracks = defaultdict(list)
        self.vehicle_boxes = defaultdict(list)
        self.vehicle_sign_history = None
        self.lane_changed_vehicles = set()
        self.frame_shape = None

    # ----------- Lane processing methods -----------
    def extract_lane_segments_all_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        segments = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = self.lane_model.predict(source=video_path, stream=True, save=False)
        from tqdm import tqdm
        for result in tqdm(results, desc="Extracting lane segments", total=total_frames):
            if result.masks is not None:
                class_names = self.lane_model.names
                for i in range(len(result.boxes)):
                    mask_tensor = result.masks.data[i]
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                    orig_h, orig_w = result.orig_shape
                    mask_resized = cv2.resize(mask_np, (orig_w, orig_h))
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) < 2:
                            continue
                        contour = contour.squeeze(1)
                        if len(contour.shape) == 1:
                            continue
                        segments.append(contour)
        cap.release()
        return segments, (H, W)

    @staticmethod
    def direction_vector(p1, p2):
        v = np.array(p2) - np.array(p1)
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    @staticmethod
    def can_merge(segA, segB, angle_thresh_deg=20, dist_thresh=40):
        a1, a2 = segA[0], segA[-1]
        b1, b2 = segB[0], segB[-1]
        pairs = [(a2, b1), (a2, b2), (a1, b1), (a1, b2)]
        for pA, pB in pairs:
            if np.linalg.norm(np.array(pA) - np.array(pB)) < dist_thresh:
                vecA = LaneChangeDetector.direction_vector(a1, a2)
                vecB = LaneChangeDetector.direction_vector(b1, b2)
                if np.linalg.norm(vecA) == 0 or np.linalg.norm(vecB) == 0:
                    continue
                angle = np.arccos(np.clip(np.dot(vecA, vecB), -1, 1)) * 180 / np.pi
                if angle < angle_thresh_deg:
                    if np.array_equal(pA, a2):
                        return 'forward', pB
                    else:
                        return 'backward', pB
        return None, None

    @classmethod
    def merge_segments(cls, segments, angle_thresh_deg=20, dist_thresh=40):
        from tqdm import tqdm
        merged = []
        used = set()
        for idx, seg in tqdm(enumerate(segments), total=len(segments), desc="Merging lane segments"):
            if idx in used:
                continue
            poly = list(seg)
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(segments):
                    if j == idx or j in used:
                        continue
                    how, _ = cls.can_merge(np.array(poly), other, angle_thresh_deg, dist_thresh)
                    if how:
                        if how == "forward":
                            poly.extend(list(other))
                        else:
                            poly = list(other) + poly
                        used.add(j)
                        changed = True
                        break
            used.add(idx)
            merged.append(np.array(poly))
        return merged

    @staticmethod
    def draw_lanes_on_frame(base_frame, polylines):
        composite = base_frame.copy()
        for i, poly in enumerate(polylines):
            color = (0, 180 + (i*35)%60, 60 + (i*25)%150)
            pts = poly.reshape((-1, 1, 2))
            cv2.polylines(composite, [pts], isClosed=False, color=color, thickness=4)
        return composite

    @staticmethod
    def signed_distance_to_polyline(pt, polyline):
        px, py = pt
        min_dist = None
        sign_at_min = 1
        for i in range(len(polyline)-1):
            x1, y1 = polyline[i]
            x2, y2 = polyline[i+1]
            dx, dy = x2-x1, y2-y1
            if dx == dy == 0:
                continue
            t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx+dy*dy)
            t = min(1, max(0, t))
            nearest_x = x1 + t*dx
            nearest_y = y1 + t*dy
            dist = np.hypot(px-nearest_x, py-nearest_y)
            vector_lane = np.array([dx, dy])
            vector_to_vehicle = np.array([px-x1, py-y1])
            cross = vector_lane[0]*vector_to_vehicle[1] - vector_lane[1]*vector_to_vehicle[0]
            sign = np.sign(cross) if cross != 0 else 1
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                sign_at_min = sign
        return sign_at_min, min_dist

    # ----------- Vehicle Tracking Subclass -----------
    class VehicleTracker:
        def __init__(self, max_disappeared=50):
            self.next_object_id = 0
            self.objects = {}
            self.disappeared = {}
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
                    iou_matrix[i, j] = LaneChangeDetector.compute_iou(obj_box, det_box[:4])
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

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(areaA + areaB - interArea + 1e-5)

    # ----------- Video Processing Methods -----------

    def detect_vehicles(self, frame):
        results = self.vehicle_model(frame, conf=self.confidence, classes=self.vehicle_classes)
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
        for object_id in self.vehicle_tracks:
            self.vehicle_boxes[object_id].append(None)
        for object_id, (x1, y1, x2, y2, conf, cls) in tracking_results:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            norm_x, norm_y = cx / width, cy / height
            self.vehicle_tracks[object_id].append((norm_x, norm_y))
            self.vehicle_boxes[object_id].append((x1, y1, x2, y2))

    def assign_lanes_and_detect_changes(self, tracking_results, frame_idx, lane_change_events, fps):
        persistence = 3
        for object_id, (x1, y1, x2, y2, conf, cls) in tracking_results:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for i, poly in enumerate(self.lane_polygons):
                sign, _ = self.signed_distance_to_polyline((cx, cy), poly)
                self.vehicle_sign_history[object_id][i].append(sign)
                signs = self.vehicle_sign_history[object_id][i]
                if len(signs) > persistence*2:
                    prev_sign = np.sign(sum(signs[-persistence*2:-persistence]))
                    new_sign = np.sign(sum(signs[-persistence:]))
                    if prev_sign != 0 and new_sign != 0 and prev_sign != new_sign:
                        already = any(e['track_id']==object_id and e['lane_idx']==i for e in lane_change_events)
                        if not already:
                            timestamp = frame_idx / fps
                            lane_change_events.append({
                                'track_id': object_id,
                                'timestamp': timestamp,
                                'frame_idx': frame_idx,
                                'lane_idx': i,
                                'direction': 'left-to-right' if prev_sign < 0 else 'right-to-left'
                            })
                            self.lane_changed_vehicles.add(object_id)

    def draw_lanes_and_regions(self, frame):
        overlay = frame.copy()
        for poly in self.lane_polygons:
            pts = poly.reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=False, color=(255,120,0), thickness=5)
            cv2.circle(overlay, tuple(pts[0][0]), 10, (0,255,0), -1)
            cv2.circle(overlay, tuple(pts[-1][0]), 10, (0,0,255), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file {video_path}")
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
        if self.vehicle_sign_history is None and self.lane_polygons:
            self.vehicle_sign_history = defaultdict(lambda: [[] for _ in range(len(self.lane_polygons))])
        try:
            from tqdm import tqdm
            pbar = tqdm(total=frame_count, desc="Processing video")
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                current_height, current_width = frame.shape[:2]
                frame_idx += 1
                frame = self.draw_lanes_and_regions(frame)
                detections = self.detect_vehicles(frame)
                tracking_results = self.tracker.update(detections)
                self.update_tracks(tracking_results, (height, width))
                self.assign_lanes_and_detect_changes(tracking_results, frame_idx, lane_change_events, fps)
                frame = self.draw_tracks(frame)
                for object_id, det in tracking_results:
                    x1, y1, x2, y2, conf, cls = det
                    color = (0, 255, 0) if object_id not in self.lane_changed_vehicles else (0, 0, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID: {object_id}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (10, 10), (370, 90), (0, 0, 0), -1)
                cv2.putText(frame, f"Frame: {frame_idx}/{frame_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Vehicles: {len(self.vehicle_tracks)}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Lane changes: {len(self.lane_changed_vehicles)}", (165, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                if show_display:
                    max_h, max_w = 900, 1600
                    if current_height > max_h or current_width > max_w:
                        scale = min(max_h / current_height, max_w / current_width)
                        display_frame = cv2.resize(frame, None, fx=scale, fy=scale)
                        cv2.imshow("Lane Change Detection", display_frame)
                    else:
                        cv2.imshow("Lane Change Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if writer and (frame_idx % save_frequency == 0):
                    writer.write(frame)
                pbar.update(1)
            pbar.close()
        finally:
            cap.release()
            if writer:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()
        return lane_change_events

    # ----------- Top-level orchestration method -----------
    @classmethod
    def run_lane_change_detection(cls, video_path, lane_model_path, vehicle_model_path,
                                  output_path=None, confidence=0.5, save_frequency=1, show_display=False):
        detector = cls(lane_model_path, vehicle_model_path, confidence=confidence)
        # 1. Lane extraction and merging
        segments, (H, W) = detector.extract_lane_segments_all_frames(video_path)
        lane_polygons = cls.merge_segments(segments, angle_thresh_deg=20, dist_thresh=40)
        detector.lane_polygons = lane_polygons
        detector.vehicle_sign_history = defaultdict(lambda: [[] for _ in range(len(lane_polygons))])
        # 2. Lane overlay save (optional)
        cap_tmp = cv2.VideoCapture(video_path)
        ret, first_frame = cap_tmp.read()
        cap_tmp.release()
        overlay_path = None
        if ret:
            overlay_path = os.path.splitext(output_path or "lane_overlay.png")[0] + "_lane_overlay.png"
            overlay_img = cls.draw_lanes_on_frame(first_frame, lane_polygons)
            cv2.imwrite(overlay_path, overlay_img)
        # 3. Main detection
        events = detector.process_video(
            video_path,
            output_path=output_path,
            show_display=show_display,
            save_frequency=save_frequency
        )
        return {
            "lane_overlay_path": overlay_path,
            "lane_count": len(lane_polygons),
            "lane_change_events": events
        }
