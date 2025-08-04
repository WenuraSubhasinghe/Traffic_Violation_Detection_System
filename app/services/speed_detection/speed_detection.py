import os
import cv2
import time
from matplotlib import patches, pyplot as plt
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
from app.database import db
from app.utils.video_converter import convert_to_browser_compatible  # MongoDB client


class SpeedViolationDetector:
    def __init__(self, model_path="models/yolov8n.pt", lane_model_path="models/lane_detection_models/lineDetection.pt"):
        self.model = YOLO(model_path)
        self.lane_model = YOLO(lane_model_path)
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
            'motorcycle': 60,
            'car': 50,
            'bus': 50,
            'truck': 40
        }
        
        self.lane_width = 3.0  # Lane width in meters for speed calculation
        self.conf_threshold = 0.2  # Confidence threshold for lane detection
        self.min_contour_area = 100  # Minimum contour area for valid lane detections
        self.outlier_threshold_kmh = 120  # Speed outlier threshold
        self.smoothing_window = 5  # Window for averaging speeds
        self.vehicle_speeds = defaultdict(list)
        self.vehicle_positions = defaultdict(list)
        self.homography_matrix = None
        self.inverse_homography = None
        self.scale_factor = None
        self.pixel_to_meter_ratio = None
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------
    # Lane Detection Functions
    # ----------------------------
    def detect_lane_elements(self, frame):
        """Detect lane boundaries using YOLOv8-seg model"""
        try:
            results = self.lane_model(frame, conf=self.conf_threshold, verbose=False)
            detections = {'lane_boundaries': []}
            class_mapping = {3: 'lane_boundary'}

            for result in results:
                boxes = result.boxes
                masks = result.masks
                if boxes is not None and masks is not None:
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if class_id != 3:  # Only process Lane_Boundary
                            continue

                        mask_tensor = masks.data[i]
                        mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                        orig_h, orig_w = frame.shape[:2]
                        mask_resized = cv2.resize(mask_np, (orig_w, orig_h))

                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contour_points = []
                        if contours:
                            valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
                            if valid_contours:
                                contour = max(valid_contours, key=cv2.contourArea).squeeze(1)
                                contour_points = [[int(point[0]), int(point[1])] for point in contour]
                            else:
                                print(f"Skipping detection: Contour area too small")

                        detection_info = {
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_mapping.get(class_id, 'Unknown'),
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'contour': contour_points
                        }

                        detections['lane_boundaries'].append(detection_info)

            return detections

        except Exception as e:
            print(f" Error in lane detection: {e}")
            return None

    def select_lane_boundaries(self, detections, frame_shape):
        """Select two lane boundaries based on horizontal position"""
        if len(detections['lane_boundaries']) < 2:
            print(" Insufficient lane boundaries: Need at least 2")
            return None, None

        boundaries = sorted(
            [d for d in detections['lane_boundaries'] if d['contour']],
            key=lambda x: np.mean([p[0] for p in x['contour']])
        )

        if len(boundaries) < 2:
            print(" Insufficient valid contours")
            return None, None

        left_boundary = boundaries[0]
        right_boundary = boundaries[-1]

        return left_boundary, right_boundary

    def calculate_reference_points(self, frame):
        """Calculate 4 reference points using lane boundaries and line intersections"""
        frame_shape = frame.shape
        height, width = frame_shape[:2]
        detections = self.detect_lane_elements(frame)

        if not detections or not detections['lane_boundaries']:
            print(" No lane boundaries detected")
            return None, None, None

        left_boundary, right_boundary = self.select_lane_boundaries(detections, frame_shape)
        if left_boundary is None or right_boundary is None:
            print(" Failed to select lane boundaries")
            return None, None, None

        # Get top and bottom points for each boundary
        left_contour = np.array(left_boundary['contour'])
        right_contour = np.array(right_boundary['contour'])

        left_y_min, left_y_max = np.min(left_contour[:, 1]), np.max(left_contour[:, 1])
        right_y_min, right_y_max = np.min(right_contour[:, 1]), np.max(right_contour[:, 1])

        left_y_top = left_y_min + 0.25 * (left_y_max - left_y_min)
        left_y_bottom = left_y_min + 0.75 * (left_y_max - left_y_min)
        right_y_top = right_y_min + 0.25 * (right_y_max - right_y_min)
        right_y_bottom = right_y_min + 0.75 * (right_y_max - right_y_min)

        def find_closest_point(contour, y_target):
            y_diffs = np.abs(contour[:, 1] - y_target)
            idx = np.argmin(y_diffs)
            return contour[idx]

        left_top = find_closest_point(left_contour, left_y_top)
        left_bottom = find_closest_point(left_contour, left_y_bottom)
        right_top = find_closest_point(right_contour, right_y_top)
        right_bottom = find_closest_point(right_contour, right_y_bottom)

        # Fit lines through top and bottom points to extend across image height
        def fit_line(top_point, bottom_point):
            x1, y1 = top_point
            x2, y2 = bottom_point
            if y1 == y2:
                return None
            m = (x2 - x1) / (y2 - y1)
            b = x1 - m * y1
            x_top = m * 0 + b
            x_bottom = m * (height - 1) + b
            return [(int(x_top), 0), (int(x_bottom), height - 1)]

        left_line = fit_line(left_top, left_bottom)
        right_line = fit_line(right_top, right_bottom)

        if left_line is None or right_line is None:
            print(" Failed to fit lines through lane boundaries")
            return None, None, None

        # Draw two horizontal lines to divide image into 3 portions
        y_third = height // 3
        horz_line1 = [(0, y_third), (width - 1, y_third)]
        horz_line2 = [(0, 2 * y_third), (width - 1, 2 * y_third)]

        # Find intersection points
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
            x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None
            
            px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
            
            if 0 <= px < width and 0 <= py < height:
                return (int(px), int(py))
            return None

        reference_points = [
            line_intersection(horz_line1, left_line),  # Top-left
            line_intersection(horz_line1, right_line),  # Top-right
            line_intersection(horz_line2, right_line),  # Bottom-right
            line_intersection(horz_line2, left_line)   # Bottom-left
        ]

        if None in reference_points:
            print(" Failed to calculate valid intersection points")
            return None, None, None

        detection_info = {
            'method': 'lane_boundaries_intersection',
            'line_types': 'lane_boundary + lane_boundary',
            'detections': detections,
            'left_boundary': left_boundary,
            'right_boundary': right_boundary,
            'lines': {
                'left_line': left_line,
                'right_line': right_line,
                'horz_line1': horz_line1,
                'horz_line2': horz_line2
            }
        }

        # Save visualization of reference points
        ref_points_image_path = self.visualize_detections(frame, detections, reference_points, detection_info['lines'])
        return reference_points, detection_info, ref_points_image_path
    
    def visualize_detections(self, frame, detections, reference_points=None, lines=None):
        """Visualize lane boundaries and reference points, save to output folder"""
        if not detections:
            print("No detections to visualize")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(frame_rgb)

        # Draw lane boundaries
        for detection in detections['lane_boundaries']:
            x1, y1, x2, y2 = detection['bbox']
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=2, edgecolor='red',
                                    facecolor='none', alpha=0.8)
            ax.add_patch(rect)

            if detection['contour']:
                contour_np = np.array(detection['contour'])
                ax.plot(contour_np[:, 0], contour_np[:, 1], color='red', linewidth=2)

            ax.text(x1, y1-5, f"{detection['class_name']}\n{detection['confidence']:.2f}",
                    color='red', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

        # Draw constructed lines
        if lines:
            for line_name, line in lines.items():
                color = 'cyan' if 'horz' in line_name else 'magenta'
                ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color=color, linewidth=2, label=line_name)

        # Draw reference points
        if reference_points:
            points = np.array(reference_points + [reference_points[0]])
            ax.plot(points[:, 0], points[:, 1], 'lime', linewidth=3, label='Reference Rectangle')

            labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
            point_colors = ['red', 'green', 'blue', 'orange']
            for i, (point, label, p_color) in enumerate(zip(reference_points, labels, point_colors)):
                ax.plot(point[0], point[1], 'o', color=p_color, markersize=10,
                        markeredgecolor='white', markeredgewidth=2)
                ax.text(point[0]+20, point[1]-20, f"{i+1}: {label}",
                        color='white', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=p_color, alpha=0.8))

        ax.set_title('Lane Boundary Detection Results', fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.legend()

        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_points_image_path = os.path.join(self.output_dir, f"reference_points_{timestamp}.png")
        plt.savefig(ref_points_image_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Reference points visualization saved: {ref_points_image_path}")
        return ref_points_image_path

    # ----------------------------
    # Vehicle Detection
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
                width = x2 - x1
                height = y2 - y1
                tracker_detections.append(([x1, y1, width, height], score, self.CLASS_NAMES[class_id]))

            # Update the tracker
            tracks = self.tracker.update_tracks(tracker_detections, frame=frame)

        return tracks
    
    # ----------------------------
    # Homography Transformation
    # ----------------------------

    def compute_homography_with_detection_info(self, reference_points, detection_info=None,
                                             bird_view_width=400, bird_view_height=600):
        dst_points = np.float32([
            [0, bird_view_height],
            [bird_view_width, bird_view_height],
            [bird_view_width, 0],
            [0, 0]
        ])

        src_points = np.float32(reference_points)
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_homography = cv2.getPerspectiveTransform(dst_points, src_points)

        bottom_width_pixels = np.linalg.norm(
            np.array(reference_points[1]) - np.array(reference_points[0])
        )

        self.pixel_to_meter_ratio = bottom_width_pixels / self.lane_width
        self.scale_factor = bird_view_width / self.lane_width

        print(f"📏 Calibration metrics:")
        print(f"   Bottom width (pixels): {bottom_width_pixels:.1f}")
        print(f"   Lane width (meters): {self.lane_width}")
        print(f"   Pixel to meter ratio: {self.pixel_to_meter_ratio:.2f}")
        print(f"   Scale factor: {self.scale_factor:.2f}")

        return self.homography_matrix

    def transform_point(self, point):
        """Transform a single point using the homography matrix"""
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not computed yet")

        point_homo = np.array([point[0], point[1], 1.0])
        transformed = self.homography_matrix @ point_homo

        if transformed[2] != 0:
            x = transformed[0] / transformed[2]
            y = transformed[1] / transformed[2]
            return (int(x), int(y))
        else:
            return None

    def transform_image(self, image, output_size=(400, 600)):
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not computed yet")
        return cv2.warpPerspective(image, self.homography_matrix, output_size)
    
    # ----------------------------
    # Distance Conversion
    # ----------------------------
    def pixels_to_meters(self, pixel_distance):
        """Convert pixel distance to meters using pixel to meter ratio"""
        if self.pixel_to_meter_ratio is None:
            raise ValueError("Pixel to meter ratio not calculated")
        return pixel_distance / self.pixel_to_meter_ratio

    # ----------------------------
    # Speed Calculation
    # ----------------------------
    def calculate_speed(self, track_id, frame_count, fps):
        """Calculate speed in km/h by tracking bounding box center across frames"""
        if not self.vehicle_positions[track_id]:
            return 0

        positions = self.vehicle_positions[track_id]
        if len(positions) < 2:
            return 0

        # Sort positions by frame number
        positions = sorted(positions, key=lambda x: x[0])
        frame1, x1, y1 = positions[-2]
        frame2, x2, y2 = positions[-1]

        # Transform points to homography-corrected space
        point1 = self.transform_point((x1, y1))
        point2 = self.transform_point((x2, y2))
        if point1 is None or point2 is None:
            return 0

        # Calculate pixel distance (Euclidean distance in transformed space)
        pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        frame_diff = frame2 - frame1
        time_diff = frame_diff / fps

        if time_diff == 0:
            return 0

        # Convert pixel distance to meters
        distance_meters = self.pixels_to_meters(pixel_distance)
        distance_km = distance_meters / 1000
        speed_kmh = (distance_km / time_diff) * 3600

        # Outlier handling
        if speed_kmh > self.outlier_threshold_kmh:
            if self.vehicle_speeds[track_id]:
                speed_kmh = np.mean(self.vehicle_speeds[track_id][-self.smoothing_window:])
            else:
                speed_kmh = 0  # If no previous speeds, use 0

        # Store speed for future averaging
        self.vehicle_speeds[track_id].append(speed_kmh)
        if len(self.vehicle_speeds[track_id]) > self.smoothing_window:
            self.vehicle_speeds[track_id] = self.vehicle_speeds[track_id][-self.smoothing_window:]

        return speed_kmh

    # ----------------------------
    # Main Processing Function
    # ----------------------------
    async def process_video(self, video_path: str, output_path: str):
        """Process video with speed violation detection using homography and bounding box center tracking"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Ensure output path is in output directory
        output_path = os.path.join(self.output_dir, os.path.basename(output_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Use original resolution

        # Tracking variables
        violations = []
        speeds = defaultdict(list)
        frame_count = 0
        start_time = time.time()

        # Initial frame processing for homography transformation
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the first frame")
            cap.release()
            return None

        # Detect lane boundaries and calculate reference points
        ref_points_image_path = None
        transformed_image_path = None
        try:
            reference_points, detection_info, ref_points_image_path = self.calculate_reference_points(frame)
            if reference_points is None:
                raise RuntimeError("Failed to detect lane boundaries for homography calibration.")

            # Compute homography matrix
            self.compute_homography_with_detection_info(reference_points, detection_info)

            # Transform and save first frame
            transformed_frame = self.transform_image(frame)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            transformed_image_path = os.path.join(self.output_dir, f"transformed_frame_{timestamp}.jpg")
            cv2.imwrite(transformed_image_path, transformed_frame)
            print(f"Transformed frame saved: {transformed_image_path}")

            if np.mean(transformed_frame) < 5:
                raise ValueError("Homography transform produced a blank or unusable frame.")

        except Exception as e:
            print(f"Automated homography calibration failed: {e}. Proceeding without transform.")
            self.homography_matrix = None
            self.inverse_homography = None
            self.pixel_to_meter_ratio = 1.0  # Fallback ratio (unreliable speed calculation)
            self.scale_factor = None

        # Reset video capture to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Main processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Use original frame for detection and annotation
            processed_frame = frame.copy()

            frame_count += 1
            if frame_count % 100 == 1:
                debug_path = os.path.join(self.output_dir, f"debug_frame_{frame_count}.jpg")
                cv2.imwrite(debug_path, processed_frame)
                print(f"Frame {frame_count} (original) saved: {debug_path}")

            # Detect and track vehicles on original frame
            detections = self.detect_vehicles(processed_frame)
            tracks = self.track_vehicles(processed_frame, detections)

            # Process each tracked vehicle
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                bbox = track.to_ltrb()
                x1, y1, x2, y2 = map(int, bbox)
                vehicle_type = track.get_det_class()
                vehicle_key = f"{vehicle_type}_{track_id}"

                # Calculate center point of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                self.vehicle_positions[track_id].append((frame_count, center_x, center_y))

                # Calculate speed
                speed = self.calculate_speed(track_id, frame_count, fps)

                speed_limit = self.speed_limits.get(vehicle_type.lower())
                if speed_limit is None:
                    print(f"[WARNING] Unknown vehicle type '{vehicle_type}'")
                    continue

                is_violation = speed > speed_limit
                if speed > 0:  # Only record non-zero speeds
                    t_finish = datetime.now()
                    violations.append((vehicle_key, speed, speed_limit, is_violation, t_finish))
                    
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
                        print(f"[DEBUG] NORMAL: {vehicle_key}")
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Default display for tracked vehicle
                else:
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

        try:
            output_path = convert_to_browser_compatible(output_path, overwrite=True)
            print(f"Converted to browser-compatible format: {output_path}")
        except Exception as e:
            print(f" Video conversion failed: {e}")

        # Print violation summary
        print("\nViolation Summary:")
        for v in violations:
            vehicle_key, speed, speed_limit, is_violation, timestamp = v
            status = "VIOLATION" if is_violation else "Normal"
            print(f"{vehicle_key} - Speed: {speed:.1f} km/h, Limit: {speed_limit} km/h, Status: {status}, Time: {timestamp}")

        # Store vehicle data in MongoDB
        vehicle_data = []
        violation_data = defaultdict(list)
        
        for vehicle_key, speed, speed_limit, is_violation, timestamp in violations:
            if is_violation:
                track_id = int(vehicle_key.split('_')[1])
                violation_data[track_id].append({
                    "time": timestamp,
                    "speed": speed,
                    "limit": speed_limit
                })

        for track_id, speed_list in speeds.items():
            vehicle_data.append({
                "vehicle_id": track_id,
                "speed_series": speed_list,
                "violations": violation_data.get(track_id, []),
                "video_path": output_path,
                "created_at": datetime.utcnow()
            })

        return {
            "summary": {
                "frame_count": frame_count,
                "processing_time": round(time.time() - start_time, 2),
                "video_path": output_path,
                "total_violations": sum(1 for v in violations if v[3]),
                "total_vehicles": len(speeds)
            },
            "vehicle_data": vehicle_data,
            "violations": violations,
            "reference_points_image": ref_points_image_path,
            "transformed_image": transformed_image_path
        }