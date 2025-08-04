from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
import json
from collections import defaultdict

class ViolationDetectionService:
    def __init__(
        self,
        vehicle_model_path: str = "models/vehicle_model/yolov8s.pt",
        sign_model_path: str = "models/sign_detection_model/best_roadsigns2.pt",
        output_dir: str = "outputs"
    ):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.sign_model = YOLO(sign_model_path)
        self.tracker = DeepSort(max_age=30)
        self.output_dir = output_dir
        self.violations_dir = os.path.join(output_dir, "violations")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.violations_dir, exist_ok=True)

    @staticmethod
    def get_sign_zone(sign_box, width, height, width_pad=600, height_pad_top=0, height_pad_bottom=600):
        x1, y1, x2, y2 = sign_box
        zone_x1 = max(0, x1 - width_pad)
        zone_x2 = min(width, x2 + width_pad)
        zone_y1 = max(0, y1 - height_pad_top)
        zone_y2 = min(height, y2 + height_pad_bottom)
        return [zone_x1, zone_y1, zone_x2, zone_y2]

    @staticmethod
    def boxes_intersect(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return (xB - xA > 0) and (yB - yA > 0)

    def process_video(self, video_path: str) -> dict:
        output_video_path = os.path.join(self.output_dir, "predicted_violation.mp4")
        violation_json_path = os.path.join(self.output_dir, "violation_ids.json")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        vehicle_tracker = defaultdict(lambda: {"frames_in_zone": 0, "violation_flagged": False})
        THRESHOLD_FRAMES = int(fps * 1.5)  # Violation if in zone > 1.5 seconds
        frame_count = 0
        violated_ids = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            sign_zones = []

            # Detect road signs
            sign_results = self.sign_model.predict(source=frame, conf=0.3, verbose=False)
            for r in sign_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Draw sign bbox for visual
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, "Stop", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    # Create detection zone around sign
                    zone = self.get_sign_zone([x1, y1, x2, y2], width, height)
                    zx1, zy1, zx2, zy2 = zone
                    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 0, 255), 2)
                    cv2.putText(frame, "sign_zone", (zx1, zy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    sign_zones.append(zone)

            # Detect vehicles
            vehicle_results = self.vehicle_model.predict(source=frame, conf=0.3, verbose=False)
            detections = []
            for r in vehicle_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.vehicle_model.model.names[cls_id]
                    if class_name in ['car', 'bus', 'truck', 'motorcycle']:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1
                        conf = float(box.conf[0])
                        detections.append(([x1, y1, w, h], conf, class_name))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                vbox = [x1, y1, x2, y2]
                # Draw vehicle bbox and id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Check violation
                for zone in sign_zones:
                    if self.boxes_intersect(vbox, zone):
                        vehicle_tracker[track_id]["frames_in_zone"] += 1
                    else:
                        vehicle_tracker[track_id]["frames_in_zone"] = 0

                    if (vehicle_tracker[track_id]["frames_in_zone"] >= THRESHOLD_FRAMES and
                        not vehicle_tracker[track_id]["violation_flagged"]):
                        vehicle_tracker[track_id]["violation_flagged"] = True
                        violated_ids.add(track_id)
                        cv2.putText(frame, "VIOLATION!", (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        # Save violation frame
                        img_path = os.path.join(self.violations_dir, f"violation_{track_id}_{frame_count}.jpg")
                        cv2.imwrite(img_path, frame)

            out.write(frame)

        # Save results
        cap.release()
        out.release()
        with open(violation_json_path, "w") as f:
            json.dump({"violating_vehicle_ids": sorted(list(violated_ids))}, f, indent=4)

        print(f"✅ Done. Outputs saved in: {self.output_dir}")
        return {
            "output_video": output_video_path,
            "violations_dir": self.violations_dir,
            "violation_json": violation_json_path,
            "violating_vehicle_ids": sorted(list(violated_ids))
        }
