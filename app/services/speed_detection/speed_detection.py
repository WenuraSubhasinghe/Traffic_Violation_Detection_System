import os
import cv2
import time
import numpy as np
import sys
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from app.database import db  # MongoDB client


class SpeedViolationDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.tracker = DeepSort(max_age=30, n_init=3)
        
        # Vehicle class IDs and names from YOLO (COCO dataset)
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.CLASS_NAMES = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Speed limits for different vehicle types (km/h)
        self.speed_limits = {
            'motorcycle': 45,
            'car': 10,
            'bus': 10,
            'truck': 10
        }
        
        self.distance_between_lines = 10.0  # meters between reference lines
        self.line1_y_percent = 0.6  # First line (bottom)
        self.line2_y_percent = 0.3  # Second line (top)
        self.reference_lines = []
        self.line_thickness = 12

    # ----------------------------
    # Affine Transformation Functions
    # ----------------------------
    def triangle_area(self, pts):
        """Calculate triangle area from 3 points"""
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = (a + b + c) / 2
        return max(np.sqrt(abs(s * (s - a) * (s - b) * (s - c))), 1e-6)

    def auto_detect_lane_landmarks(self, frame):
        """Automatically detect lane landmarks for affine transformation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Focus on lower part of image (typical location for road lanes)
        height, width = edges.shape
        mask = np.zeros_like(edges)
        roi = np.array([[
            (int(width*0.25), int(height*0.8)),
            (int(width*0.75), int(height*0.8)),
            (int(width*0.95), height-1),
            (int(width*0.05), height-1)
        ]], dtype=np.int32)
        cv2.fillPoly(mask, roi, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=100, 
                               minLineLength=80, maxLineGap=80)
        points = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                points.append([x1, y1])
                points.append([x2, y2])
                
            # Sort and select 3 representative points
            points = np.array(points)
            if len(points) < 3:
                raise RuntimeError("Not enough lane points for affine calibration.")
                
            # Choose leftmost, rightmost, and lowest points to ensure spread
            leftmost = points[np.argmin(points[:, 0])]
            rightmost = points[np.argmax(points[:, 0])]
            lowest = points[np.argmax(points[:, 1])]
            selected_pts = np.float32([leftmost, rightmost, lowest])
        else:
            raise RuntimeError("Could not automatically detect lane lines for affine calibration.")

        if len(selected_pts) < 3:
            raise RuntimeError("Detected less than three lane points for affine transformation.")

        return selected_pts  # Shape (3,2)

    def warp_frame_affine(self, frame, src_pts, dst_pts):
        """Apply affine transformation to frame"""
        M = cv2.getAffineTransform(src_pts, dst_pts)
        height, width = frame.shape[:2]
        warped = cv2.warpAffine(frame, M, (width, height))
        return warped, M

    # ----------------------------
    # Vehicle Detection with Improved Classes
    # ----------------------------
    def detect_vehicles(self, frame):
        """Detect vehicles using YOLOv8 with specific vehicle classes"""
        results = self.model.predict(source=frame, classes=self.VEHICLE_CLASSES, conf=0.3)

        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                print(f"Detected class: {class_id}")
                detections.append([x1, y1, x2, y2, confidence, class_id])

        return np.array(detections) if len(detections) > 0 else np.empty((0, 6))

    def track_vehicles(self, frame, detections):
        """Track vehicles using DeepSORT"""
        tracks = []
        if len(detections) > 0:
            bboxes = detections[:, :4]  # x1, y1, x2, y2
            scores = detections[:, 4]
            class_ids = detections[:, 5].astype(int)

            # Format detections for DeepSORT
            tracker_detections = []
            for bbox, score, class_id in zip(bboxes, scores, class_ids):
                x1, y1, x2, y2 = bbox
                tracker_detections.append(([x1, y1, x2, y2], score, self.CLASS_NAMES[class_id]))

            # Update the tracker
            tracks = self.tracker.update_tracks(tracker_detections, frame=frame)

        return tracks

    # ----------------------------
    # Reference Lines with Stripe Detection
    # ----------------------------
    def create_reference_lines(self, frame, line1_y_percent=None, line2_y_percent=None, line_thickness=None):
        """Create two reference lines for speed calculation with stripe bands"""
        if line1_y_percent is None:
            line1_y_percent = self.line1_y_percent
        if line2_y_percent is None:
            line2_y_percent = self.line2_y_percent
        if line_thickness is None:
            line_thickness = self.line_thickness
            
        height, width = frame.shape[:2]
        
        line1_y = int(height * line1_y_percent)
        line2_y = int(height * line2_y_percent)

        half_th = line_thickness // 2
        # Stripe band as [start_y, end_y] (inclusive)
        line1_band = (max(0, line1_y - half_th), min(height-1, line1_y + half_th))
        line2_band = (max(0, line2_y - half_th), min(height-1, line2_y + half_th))

        frame_with_stripes = frame.copy()
        cv2.rectangle(frame_with_stripes, (0, line1_band[0]), (width, line1_band[1]), (0, 0, 255), -1)
        cv2.rectangle(frame_with_stripes, (0, line2_band[0]), (width, line2_band[1]), (0, 255, 0), -1)

        return frame_with_stripes, line1_band, line2_band

    # ----------------------------
    # Improved Intersection Detection
    # ----------------------------
    def check_intersection(self, bbox, line_y):
        """Check if the bounding box intersects with a horizontal reference line"""
        x1, y1, x2, y2 = bbox
        return y1 <= line_y <= y2

    def check_stripe_intersection(self, bbox, band):
        """Check if bounding box intersects with stripe band (more robust)"""
        x1, y1, x2, y2 = bbox
        band_start, band_end = band
        # Check for overlap in Y
        return not (y2 < band_start or y1 > band_end)

    # ----------------------------
    # Speed Calculation
    # ----------------------------
    def calculate_speed(self, distance_meters, time_seconds):
        """Calculate speed in km/h"""
        if time_seconds == 0:
            return 0
        distance_km = distance_meters / 1000
        speed_kmh = (distance_km / time_seconds) * 3600
        return speed_kmh

    # ----------------------------
    # Main Processing Function
    # ----------------------------
    async def process_video(self, video_path: str, output_path: str):
        """Process video with comprehensive speed violation detection"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Tracking variables
        cache_matrix = []  # (start_frame, vehicle_key)
        violations = []    # (vehicle_key, speed, limit, is_violation, timestamp)
        speeds = defaultdict(list)
        frame_count = 0
        start_time = time.time()

        # Initial frame processing for affine transformation
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the first frame")
            return None

        # AUTOMATED: Detect three lane markings and define destination points
        affine_M = None
        try:
            src_pts = self.auto_detect_lane_landmarks(frame)
            triangle = self.triangle_area(src_pts)
            print(f"Selected lane points: {src_pts}, triangle area: {triangle}")
            sys.stdout.flush()
            
            # Map these to destination points
            dst_pts = np.float32([
                [0, 0],         # First landmark
                [3, 0],         # 3 meters to the right of first
                [0, 3]          # 3 meters vertically above the first
            ]) * 100  # scale for pixel space
            
            # Geometry check before applying affine
            if self.triangle_area(src_pts) < 1000:  # threshold can be adjusted
                print("Bad affine calibration: lane points too close or nearly colinear. Skipping affine for this frame.")
                sys.stdout.flush()
                affine_M = None
            else:
                frame, affine_M = self.warp_frame_affine(frame, src_pts, dst_pts)
                cv2.imwrite('debug_affine_first_frame.jpg', frame)
                print("Affine-warped frame saved as 'debug_affine_first_frame.jpg'. Please open and inspect visually.")
                sys.stdout.flush()
                if np.mean(frame) < 5:
                    raise ValueError("Affine transform produced a blank or unusable frame.")

        except Exception as e:
            print(f"Automated affine calibration failed: {e}. Proceeding without affine transform.")
            sys.stdout.flush()
            affine_M = None

        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Main processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Apply affine transformation if available
            if affine_M is not None:
                warped = cv2.warpAffine(frame, affine_M, (width, height))
                if np.mean(warped) < 5:  # Check if frame is mostly blank
                    print("Warning: Most of the affine-warped frame is blank. Reverting to original.")
                    processed_frame = frame.copy()
                else:
                    processed_frame = warped
            else:
                processed_frame = frame.copy()

            frame_count += 1
            if frame_count % 100 == 1 and affine_M is not None:
                cv2.imwrite(f'debug_affine_frame_{frame_count}.jpg', processed_frame)
                print(f"Frame {frame_count} (affine-warped) saved for review.")

            # Detect and track vehicles
            detections = self.detect_vehicles(processed_frame)
            tracks = self.track_vehicles(processed_frame, detections)

            # Create reference lines with stripes
            frame_with_stripes, line1_band, line2_band = self.create_reference_lines(processed_frame)

            # Draw reference lines on the output frame
            cv2.rectangle(processed_frame, (0, line1_band[0]), (width, line1_band[1]), (0, 0, 255), -1)
            cv2.rectangle(processed_frame, (0, line2_band[0]), (width, line2_band[1]), (0, 255, 0), -1)

            # Process each tracked vehicle
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                bbox = track.to_ltrb()
                x1, y1, x2, y2 = map(int, bbox)
                vehicle_type = track.get_det_class()
                vehicle_key = f"{vehicle_type}_{track_id}"

                # START line detection (line1)
                if self.check_stripe_intersection([x1, y1, x2, y2], line1_band):
                    if vehicle_key not in [v[1] for v in cache_matrix]:
                        cache_matrix.append((frame_count, vehicle_key))
                        print(f"[DEBUG] START crossed: {vehicle_key} at frame {frame_count}")

                # FINISH line detection and speed calculation (line2)
                if self.check_stripe_intersection([x1, y1, x2, y2], line2_band):
                    start_frame = next((f for f, k in cache_matrix if k == vehicle_key), None)
                    if start_frame is not None:
                        frame_diff = frame_count - start_frame
                        time_diff = frame_diff / fps
                        speed = self.calculate_speed(self.distance_between_lines, time_diff)

                        speed_limit = self.speed_limits.get(vehicle_type.lower())
                        if speed_limit is None:
                            print(f"[WARNING] Unknown vehicle type '{vehicle_type}'")
                            continue

                        is_violation = speed > speed_limit
                        t_finish = datetime.now()
                        violations.append((vehicle_key, speed, speed_limit, is_violation, t_finish))
                        
                        # Store speed data
                        speeds[track_id].append({
                            "time": frame_count / fps,
                            "speed": speed,
                            "vehicle_type": vehicle_type
                        })

                        speed_text = f"{vehicle_type} ID:{track_id} - {speed:.1f} km/h"
                        cv2.putText(processed_frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if is_violation:
                            print(f"[DEBUG] 🚨 VIOLATION: {vehicle_key}")
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(processed_frame, f"VIOLATION! Limit: {speed_limit} km/h", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            print(f"[DEBUG] ✅ NORMAL: {vehicle_key}")
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Remove from cache after processing
                        cache_matrix = [(f, k) for f, k in cache_matrix if k != vehicle_key]

                # Default display for tracked vehicle (if not yet processed)
                if vehicle_key not in [v[0] for v in violations]:
                    cv2.putText(processed_frame, f"{vehicle_type} ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Debug info overlay
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(processed_frame, f"Vehicles: {len(tracks)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(processed_frame, f"Violations: {sum(1 for v in violations if v[3])}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(processed_frame)

        cap.release()
        out.release()

        # Print violation summary
        print("\nViolation Summary:")
        for v in violations:
            vehicle_key, speed, speed_limit, is_violation, timestamp = v
            status = "VIOLATION" if is_violation else "Normal"
            print(f"{vehicle_key} - Speed: {speed:.1f} km/h, Limit: {speed_limit} km/h, Status: {status}, Time: {timestamp}")

        # Store vehicle data in MongoDB
        vehicle_data = []
        violation_data = defaultdict(list)
        
        # Group violations by vehicle
        for vehicle_key, speed, speed_limit, is_violation, timestamp in violations:
            if is_violation:
                track_id = int(vehicle_key.split('_')[1])
                violation_data[track_id].append({
                    "time": timestamp,
                    "speed": speed,
                    "limit": speed_limit
                })

        # Prepare data for database
        for track_id, speed_list in speeds.items():
            vehicle_data.append({
                "vehicle_id": track_id,
                "speed_series": speed_list,
                "violations": violation_data.get(track_id, []),
                "video_path": output_path,
                "created_at": datetime.utcnow()
            })

        if vehicle_data:
            await db.speed_records.insert_many(vehicle_data)

        return {
            "summary": {
                "frame_count": frame_count,
                "processing_time": round(time.time() - start_time, 2),
                "video_path": output_path,
                "total_violations": sum(1 for v in violations if v[3]),
                "total_vehicles": len(speeds)
            },
            "vehicle_data": vehicle_data,
            "violations": violations
        }