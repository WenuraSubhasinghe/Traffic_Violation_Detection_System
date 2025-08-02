# Updated main execution code with direct file browser (no webcam option)
from torch import empty, any as there
from datetime import datetime
import sys
import cv2
from cv2 import VideoCapture, resize
import math
import os
from pathlib import Path

from deep_sort_realtime.deepsort_tracker import DeepSort

from app.services.light_violation_detection.annotate import annotate
from app.services.light_violation_detection.config import CLASSES
from app.services.light_violation_detection.enhanced_violation_detector import EnhancedViolationDetector
from app.services.light_violation_detection.meta_model_manager import MetaModelManager
from app.services.light_violation_detection.stop_line_detector import EnhancedStopLineDetector
from app.services.light_violation_detection.colorRecog import recognize_color, chooseOne
from app.utils.video_converter import convert_to_browser_compatible

# Enhanced logger with violation logging
class EnhancedLogger:
    def __init__(self):
        self.violations = []
        self.frame_count = 0
        self.logged_violators = set()
        self.violation_cooldown = {}

    def update_frame_count(self):
        self.frame_count += 1

    def log_violation(self, frame_number, vehicle_data, detection_method, **kwargs):
        vehicle_id = vehicle_data[-1]  # Last element is DeepSORT track_id
        if vehicle_id in self.logged_violators:
            return False
        violation_data = {
            'frame_number': frame_number,
            'vehicle_id': vehicle_id,
            'vehicle_data': vehicle_data,
            'detection_method': detection_method,
            **kwargs
        }
        self.violations.append(violation_data)
        self.violation_cooldown[vehicle_id] = frame_number
        self.logged_violators.add(vehicle_id)
        print(f"🚨 VIOLATION: Frame {frame_number}, Vehicle ID: {vehicle_id}, Method: {detection_method}")
        return True

    def reset_violated_vehicles(self):
        pass

    def get_unique_violators(self):
        return len(self.logged_violators)

    def print_summary(self):
        unique_violators = self.get_unique_violators()
        primary_violations = sum(1 for v in self.violations if 'primary' in v.get('detection_method', ''))
        fallback_violations = sum(1 for v in self.violations if 'fallback' in v.get('detection_method', ''))
        print("=" * 80)
        print("ENHANCED STOP LINE DETECTION SUMMARY REPORT")
        print("=" * 80)
        print(f"Total Frames Processed: {self.frame_count}")
        print(f"Total Violation Events: {len(self.violations)}")
        print(f"Unique Violating Vehicles: {unique_violators}")
        print(f"Primary Detection Method Violations: {primary_violations}")
        print(f"Fallback Detection Method Violations: {fallback_violations}")
        if self.frame_count > 0:
            print(f"Unique Violation Rate: {(unique_violators/self.frame_count)*100:.4f}%")
        print("=" * 80)

def select_video_file():
    """
    Direct file browser to select video file
    """
    print("\n" + "="*60)
    print("🎥 ENHANCED TRAFFIC VIOLATION DETECTION SYSTEM")
    print("="*60)
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        print("🗂️ Opening file browser to select video file...")
        
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)  # Bring file dialog to front
        
        # File dialog with video file filters
        file_path = filedialog.askopenfilename(
            title="Select Video File for Traffic Analysis",
            initialdir=os.getcwd(),  # Start in current directory
            filetypes=[
                ("All Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v *.3gp"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
                ("WMV files", "*.wmv"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()  # Clean up
        
        if file_path:
            print(f"✅ Selected: {os.path.basename(file_path)}")
            print(f"📁 Full path: {file_path}")
            return file_path
        else:
            print("❌ No file selected. Exiting...")
            sys.exit(0)
            
    except ImportError:
        print("❌ tkinter not available on your system.")
        print("Fallback: Please install tkinter for GUI file browser.")
        
        # Fallback to manual input
        print("Enter video file path manually:")
        fallback_path = input("Video file path: ").strip().strip('"').strip("'")
        if os.path.exists(fallback_path) and os.path.isfile(fallback_path):
            return fallback_path
        else:
            print("❌ Invalid file path. Exiting...")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error opening file browser: {e}")
        sys.exit(1)

def validate_video_file(video_path):
    """
    Validate if the video file can be opened and read
    """
    if not os.path.exists(video_path):
        return False, "File does not exist"
    
    if not os.path.isfile(video_path):
        return False, "Path is not a file"
    
    cap = VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file - unsupported format or corrupted file"
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Cannot read video frames - file may be corrupted"
    
    return True, "Video file is valid"

def run_light_violation_detection(video_path):
    """
    Run the complete light violation detector system on the given video file.
    Returns summary and output video path.
    """
    # Initialize enhanced components
    model_manager = MetaModelManager()
    violation_detector = EnhancedViolationDetector()
    enhanced_stop_line_detector = EnhancedStopLineDetector()
    logger = EnhancedLogger()

    # Validate video file
    is_valid, message = validate_video_file(video_path)
    if not is_valid:
        print(f"❌ Video validation failed: {message}")
        return {"success": False, "message": message}

    cap = VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video file.")
        return {"success": False, "message": "Failed to open video file."}

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / video_fps if video_fps > 0 else 0

    success, frame = cap.read()
    if not success:
        print("❌ Failed to read the first frame.")
        return {"success": False, "message": "Failed to read the first frame."}

    if frame.shape[0] > 720:
        scale = 720 / frame.shape[0]
        frame = resize(frame, None, fx=scale, fy=scale)
    frame_height, frame_width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    input_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f'enhanced_{input_name}_output.mp4'
    output_path = os.path.join(output_folder, output_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print("❌ Failed to initialize video writer.")
        return {"success": False, "message": "Failed to initialize video writer."}

    # ===== Initialize DeepSORT tracker =====
    tracker = DeepSort(
        max_age=30,
        n_init=2,
        nms_max_overlap=1.0,
        embedder="mobilenet",
        half=True
    )

    frame_number = 1
    logger = EnhancedLogger()
    logger.update_frame_count()
    previous_light_color = None

    out.write(frame)
    position_history = {}
    frame_confirmations = {}

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1
            logger.update_frame_count()

            if frame.shape[0] > 720:
                scale = 720 / frame.shape[0]
                frame = resize(frame, None, fx=scale, fy=scale)

            current_model, model_type, adaptive_confidence, conditions = model_manager.get_optimal_model_and_settings(frame)
            detection_confidence = 0.4

            detection_result = current_model.track(
                frame,
                persist=True,
                classes=[n for n in CLASSES],
                conf=detection_confidence
            )[0].boxes.data

            traffic_lights = detection_result[detection_result[:, -1] == 9]
            vehicles = detection_result[detection_result[:, -1] != 9]

            light_colors = recognize_color(frame, traffic_lights, print_info=False)
            chosen = chooseOne(light_colors)
            has_red_light = len(light_colors['red']) > 0
            current_light_color = "red" if has_red_light else "green/yellow"

            if previous_light_color == "red" and current_light_color == "green/yellow":
                enhanced_stop_line_detector.reset_for_new_light_cycle()

            annotate(frame, traffic_lights, recognizedColor=light_colors)

            if has_red_light and chosen[0] >= 0 and len(traffic_lights) > 0:
                chosen_light_bbox = traffic_lights[chosen[0]][:4].cpu().numpy().astype(int)

                violation_boundary, detection_method = enhanced_stop_line_detector.establish_persistent_stop_line(
                    frame, chosen_light_bbox, current_light_color
                )

                frame = enhanced_stop_line_detector.visualize_detection_results(
                    frame, violation_boundary, detection_method, chosen_light_bbox
                )

                if violation_boundary is not None:
                    valid_boxes = []
                    valid_confidences = []
                    valid_classes = []
                    for vehicle in vehicles:
                        arr = vehicle[:6].cpu().numpy().tolist()
                        if len(arr) != 6 or not all(isinstance(v, (int, float)) for v in arr):
                            continue
                        x1, y1, x2, y2, conf, cls = arr
                        w = x2 - x1
                        h = y2 - y1
                        if w <= 0 or h <= 0 or math.isnan(w) or math.isnan(h):
                            continue
                        cx = x1 + w / 2
                        cy = y1 + h / 2
                        box = [float(cx), float(cy), float(w), float(h)]
                        valid_boxes.append(box)
                        valid_confidences.append(float(conf))
                        valid_classes.append(int(cls))

                    tracks = []
                    if valid_boxes:
                        detections = []
                        for box, conf, cls in zip(valid_boxes, valid_confidences, valid_classes):
                            detections.append((box, conf, cls))

                        if detections:
                            tracks = tracker.update_tracks(detections, frame=frame)
                        else:
                            tracks = []

                    violating_vehicles = []

                    for track in tracks:
                        if not track.is_confirmed():
                            continue
                        ltrb = track.to_ltrb()
                        track_id = track.track_id

                        is_violation = violation_detector.check_vehicle_violation(
                            vehicle_bbox=ltrb,
                            violation_boundary=violation_boundary,
                            vehicle_id=track_id,
                            previous_positions=position_history,
                            frame_confirmations=frame_confirmations,
                            required_frames=1
                        )
                        if is_violation:
                            violating_vehicles.append(list(ltrb) + [track_id])
                            logger.log_violation(
                                frame_number=frame_number,
                                timestamp=datetime.now(),
                                vehicle_data=list(ltrb) + [track_id],
                                detection_method=detection_method,
                                confidence=chosen[1]
                            )

                    frame = violation_detector.visualize_violation_detection(
                        frame, chosen_light_bbox, violation_boundary, detection_method, violating_vehicles
                    )

            previous_light_color = current_light_color

            out.write(frame)

    except KeyboardInterrupt:
        print("\n⚠️ Enhanced detection interrupted by user")

    finally:
        cap.release()
        out.release()
        output_path = convert_to_browser_compatible(output_path, overwrite=True)
        logger.print_summary()
        print(f"\n✅ Video processing completed!")
        print(f"📁 Output saved as: {output_path}")

    return {
        "success": True,
        "output_path": output_path,
        "summary": {
            "total_frames": logger.frame_count,
            "total_violations": len(logger.violations),
            "unique_violators": logger.get_unique_violators(),
            "violations": logger.violations
        }
    }
