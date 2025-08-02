import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import math
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

class EnhancedStopLineDetector:
    def __init__(self, model_path: str = "model/lane_detection_models/lineDetection.pt"):
        # Initialize YOLOv8 model for primary detection
        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model.fuse()  # Fuse layers for faster inference
            
            # Set model parameters explicitly for better stop line detection
            self.yolo_model.overrides = {
                'verbose': False,  # Reduce verbose output
                'save': False,    # Don't save intermediate results
                'save_txt': False,
                'save_conf': True,
                'conf': 0.15,      # Lower confidence threshold
                'iou': 0.4,        # Lower IoU for better separation
                'max_det': 15,     # Allow more detections
                'augment': True,   # Test-time augmentation
                'agnostic_nms': False,  # Class-aware NMS
            }
            
            # Optimize model for stop line detection
            self.optimize_model_for_stop_lines()
            
            print(f"YOLOv8 model loaded and optimized from {model_path}")
        except Exception as e:
            print(f"Failed to load YOLOv8 model: {e}")
            self.yolo_model = None
        
        # Parameters for stop line detection
        self.canny_low = 30
        self.canny_high = 100
        self.hough_threshold = 50
        self.min_line_length = 30
        self.max_line_gap = 20
        
        # Parameters for pole detection
        self.pole_search_width = 100
        self.pole_min_height = 50
        self.pole_base_offset = 10  # Fixed distance from pole base to stop line (positive = above pole base)
        
        # Stop line validation parameters
        self.detection_confidence_threshold = 0.15  # Lower threshold for better detection
        self.STOP_LINE_CLASS_ID = 8
        
        # Enhanced detection parameters
        self.use_multi_approach = True  # Enable multiple detection methods
        self.enable_preprocessing = True  # Enable enhanced preprocessing
        
        # **Persistence Management**
        self.current_light_state = None
        self.fixed_stop_line = None
        self.stop_line_established = False
        self.detection_method_used = None
        self.last_valid_stop_line = None
        
        # **NEW: Pole Base Persistence Management**
        self.fixed_pole_base = None
        self.pole_base_established = False
        self.pole_detection_attempts = 0
        self.max_pole_detection_attempts = 5  # Try pole detection for first 5 frames
        self.pole_candidates = []  # Store multiple pole candidates to find the lowest
        
        print("✅ Enhanced Stop Line Detector initialized with persistence and optimizations")

    def optimize_model_for_stop_lines(self):
        """Optimize model parameters specifically for stop line detection"""
        if self.yolo_model is not None:
            # Warmup the model
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.yolo_model(dummy_img, verbose=False)  # Warmup
            print("🔧 Model optimized and warmed up for stop line detection")

    def reset_for_new_light_cycle(self):
        """Reset stop line AND pole base when light changes from red to green"""
        self.fixed_stop_line = None
        self.stop_line_established = False
        self.detection_method_used = None
        
        # **NEW: Reset pole base persistence**
        self.fixed_pole_base = None
        self.pole_base_established = False
        self.pole_detection_attempts = 0
        self.pole_candidates = []
        
        print("🔄 Stop line and pole base reset for new light cycle")

    def preprocess_frame_for_detection(self, frame: np.ndarray) -> np.ndarray:
        """Apply preprocessing that matches training conditions"""
        try:
            # 1. Ensure proper image format and size
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB if needed (YOLOv8 expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # 2. Apply contrast and brightness enhancement
            enhanced = cv2.convertScaleAbs(frame_rgb, alpha=1.2, beta=10)
            
            # 3. Noise reduction while preserving edges
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # 4. Histogram equalization for better contrast
            if len(denoised.shape) == 3:
                lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
                enhanced_final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                enhanced_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
            
            return enhanced_final
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return frame

    def run_yolo_detection(self, frame: np.ndarray, method_name: str) -> List[Dict[str, Any]]:
        """Run YOLO detection with detailed logging"""
        try:
            if self.yolo_model is None:
                return []
                
            results = self.yolo_model(
                frame,
                conf=0.1,  # Very low threshold to catch all possibilities
                iou=0.5,
                max_det=20,
                classes=[self.STOP_LINE_CLASS_ID],
                verbose=False,
                augment=True,
                half=False
            )
            
            detections = []
            for result in results:
                # Check for segmentation masks first
                if result.masks is not None and result.boxes is not None:
                    masks = result.masks.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for i, (mask, box) in enumerate(zip(masks, boxes)):
                        if len(box) >= 6:
                            class_id = int(box[5])
                            confidence = float(box[4])
                            
                            if class_id == self.STOP_LINE_CLASS_ID and confidence >= 0.1:
                                polygon_coords = self.extract_polygon_from_mask(mask, frame.shape[:2])
                                
                                if polygon_coords is not None:
                                    detection = {
                                        'method': method_name,
                                        'id': i,
                                        'confidence': confidence,
                                        'bbox': box[:4].tolist(),
                                        'polygon': polygon_coords,
                                        'center_y': np.mean([point[1] for point in polygon_coords])
                                    }
                                    detections.append(detection)
                                    print(f"   ✅ {method_name} (mask): Conf={confidence:.3f}")
                
                # Check for regular bounding box detections
                elif result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        if len(box) >= 6:
                            class_id = int(box[5])
                            confidence = float(box[4])
                            
                            if class_id == self.STOP_LINE_CLASS_ID and confidence >= 0.1:
                                detection = {
                                    'method': method_name,
                                    'id': i,
                                    'confidence': confidence,
                                    'bbox': box[:4].tolist(),
                                    'polygon': self.bbox_to_polygon(box[:4]),
                                    'center_y': (box[1] + box[3]) / 2
                                }
                                detections.append(detection)
                                print(f"   ✅ {method_name} (bbox): Conf={confidence:.3f}")
            
            return detections
        except Exception as e:
            print(f"Detection failed for {method_name}: {e}")
            return []

    def bbox_to_polygon(self, bbox):
        """Convert bounding box to polygon format"""
        x1, y1, x2, y2 = bbox
        return [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]

    def filter_and_merge_detections(self, all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and merge multiple detections to get best results"""
        if not all_detections:
            return []
        
        # Sort by confidence
        sorted_detections = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates based on spatial overlap
        filtered_detections = []
        for detection in sorted_detections:
            is_duplicate = False
            for existing in filtered_detections:
                # Check if centers are too close (indicating same line)
                center_dist = abs(detection['center_y'] - existing['center_y'])
                if center_dist < 20:  # Pixels threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        print(f"Filtered detections: {len(filtered_detections)} from {len(all_detections)} candidates")
        return filtered_detections[:3]  # Return top 3 detections

    def primary_stop_line_detection_multi_approach(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Try multiple detection approaches for better results"""
        
        all_detections = []
        
        # Approach 1: Original frame
        detections_1 = self.run_yolo_detection(frame, "original_frame")
        all_detections.extend(detections_1)
        
        # Approach 2: Enhanced frame
        if self.enable_preprocessing:
            enhanced_frame = self.preprocess_frame_for_detection(frame)
            detections_2 = self.run_yolo_detection(enhanced_frame, "enhanced_frame")
            all_detections.extend(detections_2)
        
        # Approach 3: Multiple scales
        for scale in [0.8, 1.2]:
            try:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_frame = cv2.resize(frame, (new_w, new_h))
                detections_scaled = self.run_yolo_detection(scaled_frame, f"scale_{scale}")
                
                # Rescale coordinates back to original frame
                for det in detections_scaled:
                    det['bbox'] = [coord / scale for coord in det['bbox']]
                    det['polygon'] = [(int(x/scale), int(y/scale)) for x, y in det['polygon']]
                    det['center_y'] = det['center_y'] / scale
                
                all_detections.extend(detections_scaled)
            except Exception as e:
                print(f"Scale {scale} detection failed: {e}")
        
        # Remove duplicates and return best detections
        return self.filter_and_merge_detections(all_detections)

    def primary_stop_line_detection_single(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Single approach primary detection (fallback method)"""
        return self.run_yolo_detection(frame, "single_approach")

    def primary_stop_line_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Primary Detection using enhanced multi-approach method"""
        if self.yolo_model is None:
            return []
        
        if self.use_multi_approach:
            return self.primary_stop_line_detection_multi_approach(frame)
        else:
            return self.primary_stop_line_detection_single(frame)

    def extract_polygon_from_mask(self, mask: np.ndarray, frame_shape: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Extract polygon coordinates from segmentation mask"""
        try:
            mask_resized = cv2.resize(mask, (frame_shape[1], frame_shape[0]))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            polygon_points = [(int(point[0][0]), int(point[0][1])) for point in polygon]
            
            return polygon_points
            
        except Exception as e:
            print(f"Polygon extraction failed: {e}")
            return None

    def detect_traffic_light_pole_persistent(self, frame: np.ndarray, traffic_light_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Enhanced pole detection with persistence - once detected, it stays fixed"""
        
        # If pole base already established, return it
        if self.pole_base_established and self.fixed_pole_base is not None:
            print(f"Using established pole base: {self.fixed_pole_base}")
            return self.fixed_pole_base
        
        # If we've exceeded max attempts, stop trying and use best candidate
        if self.pole_detection_attempts >= self.max_pole_detection_attempts:
            if self.pole_candidates:
                # Find the lowest (highest Y value) pole candidate
                lowest_pole = max(self.pole_candidates, key=lambda p: p[1])
                self.fixed_pole_base = lowest_pole
                self.pole_base_established = True
                print(f"Pole base ESTABLISHED from candidates (lowest): {self.fixed_pole_base}")
                return self.fixed_pole_base
            else:
                print("No pole candidates found after max attempts")
                return None
        
        # Increment detection attempts
        self.pole_detection_attempts += 1
        print(f"🔍 Pole detection attempt {self.pole_detection_attempts}/{self.max_pole_detection_attempts}")
        
        # Run pole detection
        detected_pole = self.detect_traffic_light_pole_single_frame(frame, traffic_light_bbox)
        
        if detected_pole is not None:
            # Add to candidates list
            self.pole_candidates.append(detected_pole)
            print(f"Pole candidate found: {detected_pole} (Total candidates: {len(self.pole_candidates)})")
            
            # If we have enough candidates or high confidence, establish immediately
            if len(self.pole_candidates) >= 3:
                # Find the lowest (most grounded) pole position
                lowest_pole = max(self.pole_candidates, key=lambda p: p[1])
                self.fixed_pole_base = lowest_pole
                self.pole_base_established = True
                print(f"✅ Pole base ESTABLISHED early (lowest from {len(self.pole_candidates)} candidates): {self.fixed_pole_base}")
                return self.fixed_pole_base
        
        # Return current best candidate or None
        if self.pole_candidates:
            return max(self.pole_candidates, key=lambda p: p[1])  # Return lowest for now
        return None

    def detect_traffic_light_pole_single_frame(self, frame: np.ndarray, traffic_light_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Original pole detection logic for single frame (renamed for clarity)"""
        tl_x1, tl_y1, tl_x2, tl_y2 = traffic_light_bbox
        frame_height, frame_width = frame.shape[:2]
        
        tl_center_x = (tl_x1 + tl_x2) // 2
        tl_center_y = (tl_y1 + tl_y2) // 2
        
        search_x1 = max(0, tl_center_x - self.pole_search_width // 2)
        search_x2 = min(frame_width, tl_center_x + self.pole_search_width // 2)
        search_y1 = tl_y2
        search_y2 = min(frame_height, tl_y2 + 500)
        
        search_roi = frame[search_y1:search_y2, search_x1:search_x2]
        gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
        
        filtered = cv2.bilateralFilter(gray_roi, 9, 75, 75)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        enhanced = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, vertical_kernel)
        
        edges1 = cv2.Canny(enhanced, 50, 150)
        edges2 = cv2.Canny(enhanced, 30, 100)
        edges_combined = cv2.bitwise_or(edges1, edges2)
        
        lines = cv2.HoughLinesP(
            edges_combined, rho=1, theta=np.pi/180, threshold=20,
            minLineLength=self.pole_min_height, maxLineGap=15
        )
        
        if lines is None:
            return None
        
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / math.pi)
            
            if 70 <= angle <= 110:
                orig_x1 = x1 + search_x1
                orig_y1 = y1 + search_y1
                orig_x2 = x2 + search_x1
                orig_y2 = y2 + search_y1
                
                line_center_x = (orig_x1 + orig_x2) / 2
                if abs(line_center_x - tl_center_x) < self.pole_search_width // 2:
                    line_length = math.sqrt((orig_x2 - orig_x1)**2 + (orig_y2 - orig_y1)**2)
                    # **ENHANCED: Prioritize lines that extend lower (higher Y values)**
                    bottom_y = max(orig_y1, orig_y2)
                    vertical_lines.append((orig_x1, orig_y1, orig_x2, orig_y2, line_length, bottom_y))
        
        if not vertical_lines:
            return None
        
        # **IMPROVED: Select line with lowest bottom point (highest Y), then by length**
        best_line = max(vertical_lines, key=lambda line: (line[5], line[4]))  # First by bottom_y, then by length
        x1, y1, x2, y2, _, bottom_y = best_line
        
        pole_base_x = int((x1 + x2) / 2)
        pole_base_y = bottom_y  # Use the actual bottom point
        
        return (pole_base_x, pole_base_y)

    def detect_traffic_light_pole(self, frame: np.ndarray, traffic_light_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Main pole detection method - now uses persistent logic"""
        return self.detect_traffic_light_pole_persistent(frame, traffic_light_bbox)

    def establish_persistent_stop_line(self, frame: np.ndarray, 
                                     traffic_light_bbox: Tuple[int, int, int, int],
                                     light_state: str) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """
        Establish and maintain a persistent stop line for the red light period
        """
        # Handle light state changes
        if self.current_light_state != light_state:
            if self.current_light_state == "red" and light_state != "red":
                # Light changed from red to non-red, reset stop line AND pole base
                self.reset_for_new_light_cycle()
            self.current_light_state = light_state

        # If light is not red, no stop line needed
        if light_state != "red":
            return None, "no_red_light"

        # If stop line already established for this red light period, return it
        if self.stop_line_established and self.fixed_stop_line is not None:
            print(f"Using established stop line (method: {self.detection_method_used})")
            return self.fixed_stop_line, self.detection_method_used

        # Light just turned red or stop line not yet established - detect it once
        print("Red light detected - Establishing persistent stop line...")
        print(f"Frame shape: {frame.shape}, TL bbox: {traffic_light_bbox}")
        
        # **ENHANCED: Try to establish pole base first for more reliable stop line positioning**
        pole_base = self.detect_traffic_light_pole(frame, traffic_light_bbox)
        if pole_base:
            print(f"Pole base detected/used: {pole_base}")
        
        # Step 1: Enhanced Primary Detection using multiple approaches
        if self.use_multi_approach:
            detected_stop_lines = self.primary_stop_line_detection_multi_approach(frame)
        else:
            detected_stop_lines = self.primary_stop_line_detection_single(frame)
        
        if detected_stop_lines:
            print(f"Primary detection SUCCESS: Found {len(detected_stop_lines)} stop line(s)")
            
            # Log detection details
            for i, sl in enumerate(detected_stop_lines):
                print(f"   Stop line {i}: Method={sl.get('method', 'N/A')}, Conf={sl['confidence']:.3f}, Y={sl['center_y']:.1f}")
            
            # Step 2A: Multiple stop lines detected
            if len(detected_stop_lines) > 1:
                result = self.validate_multiple_stop_lines(frame, detected_stop_lines, traffic_light_bbox)
                if result is not None:
                    self.fixed_stop_line = result
                    self.detection_method_used = "primary_multiple_validated"
                    self.stop_line_established = True
                    self.last_valid_stop_line = result
                    print(f"Stop line ESTABLISHED using: {self.detection_method_used}")
                    return result, self.detection_method_used
            
            # Step 2B: Single stop line detected
            elif len(detected_stop_lines) == 1:
                result = self.validate_single_stop_line(frame, detected_stop_lines[0], traffic_light_bbox)
                self.fixed_stop_line = result
                self.detection_method_used = "primary_single_validated"
                self.stop_line_established = True
                self.last_valid_stop_line = result
                print(f"Stop line ESTABLISHED using: {self.detection_method_used}")
                return result, self.detection_method_used
        else:
            print("Primary detection FAILED - No stop lines detected by YOLOv8")
        
        # Step 3: Fallback detection when primary method fails
        print("Using fallback detection method...")
        result = self.fallback_offset_detection(frame, traffic_light_bbox)
        self.fixed_stop_line = result
        self.detection_method_used = "fallback_offset"
        self.stop_line_established = True
        self.last_valid_stop_line = result
        print(f"Stop line ESTABLISHED using: {self.detection_method_used}")
        
        return result, self.detection_method_used

    def get_current_stop_line(self) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
        """Get the currently established stop line without recalculating"""
        if self.stop_line_established and self.fixed_stop_line is not None:
            return self.fixed_stop_line, self.detection_method_used
        return None, "not_established"

    def validate_multiple_stop_lines(self, frame: np.ndarray, stop_lines: List[Dict[str, Any]], 
                                   traffic_light_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """2A. Stop Line Validation when multiple stop lines are detected"""
        if len(stop_lines) <= 1:
            return None
        
        print(f"🔍 Multiple stop lines detected ({len(stop_lines)}), validating with pole detection...")
        
        pole_base = self.detect_traffic_light_pole(frame, traffic_light_bbox)
        
        if pole_base is None:
            print("No pole detected, using closest stop line to traffic light")
            tl_center_y = (traffic_light_bbox[1] + traffic_light_bbox[3]) // 2
            closest_stop_line = min(stop_lines, key=lambda sl: abs(sl['center_y'] - tl_center_y))
            return self.convert_polygon_to_line(closest_stop_line['polygon'])
        
        pole_x, pole_y = pole_base
        frame_width = frame.shape[1]
        
        stop_line_y = pole_y - self.pole_base_offset
        line_start_x = max(0, int(frame_width * 0.15))
        line_end_x = min(frame_width, int(frame_width * 0.85))
        
        print(f"Pole-based stop line positioned at y={stop_line_y}")
        return (line_start_x, stop_line_y, line_end_x, stop_line_y)

    def validate_single_stop_line(self, frame: np.ndarray, stop_line: Dict[str, Any], 
                                traffic_light_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """2B. Validate single stop line and perform pole analysis"""
        print("🔍 Single stop line detected, performing validation...")
        
        detected_line = self.convert_polygon_to_line(stop_line['polygon'])
        pole_base = self.detect_traffic_light_pole(frame, traffic_light_bbox)
        
        if pole_base is not None:
            pole_x, pole_y = pole_base
            stop_line_y = detected_line[1]
            distance_from_pole = abs(stop_line_y - pole_y)
            
            if distance_from_pole > 100:
                print("Detected stop line seems incorrect based on pole position, using pole-based positioning")
                frame_width = frame.shape[1]
                corrected_y = pole_y - self.pole_base_offset
                line_start_x = max(0, int(frame_width * 0.15))
                line_end_x = min(frame_width, int(frame_width * 0.85))
                return (line_start_x, corrected_y, line_end_x, corrected_y)
        
        print("Single stop line validated")
        return detected_line

    def fallback_offset_detection(self, frame: np.ndarray, 
                                traffic_light_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """3. Fallback Offset module for scenarios where stop line isn't detected properly"""
        print("Activating fallback detection method...")
        
        frame_height, frame_width = frame.shape[:2]
        tl_x1, tl_y1, tl_x2, tl_y2 = traffic_light_bbox
        
        pole_base = self.detect_traffic_light_pole(frame, traffic_light_bbox)
        
        if pole_base is not None:
            print("Pole detected in fallback mode, positioning stop line")
            pole_x, pole_y = pole_base
            stop_line_y = pole_y - self.pole_base_offset
            line_start_x = max(0, int(frame_width * 0.15))
            line_end_x = min(frame_width, int(frame_width * 0.85))
            return (line_start_x, stop_line_y, line_end_x, stop_line_y)
        
        print("No pole detected, using traffic light reference positioning")
        estimated_road_y = min(frame_height - 50, tl_y2 + 200)
        line_start_x = max(0, int(frame_width * 0.1))
        line_end_x = min(frame_width, int(frame_width * 0.9))
        
        return (line_start_x, estimated_road_y, line_end_x, estimated_road_y)

    def convert_polygon_to_line(self, polygon: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Convert polygon coordinates to a horizontal line representation"""
        if not polygon:
            return (0, 0, 0, 0)
        
        avg_y = int(np.mean([point[1] for point in polygon]))
        x_coords = [point[0] for point in polygon]
        min_x = min(x_coords)
        max_x = max(x_coords)
        
        return (min_x, avg_y, max_x, avg_y)

    def visualize_detection_results(self, frame: np.ndarray, stop_line: Optional[Tuple[int, int, int, int]], 
                                  detection_method: str, traffic_light_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Enhanced visualization with pole base status"""
        vis_frame = frame.copy()
        
        tl_x1, tl_y1, tl_x2, tl_y2 = traffic_light_bbox
        cv2.rectangle(vis_frame, (tl_x1, tl_y1), (tl_x2, tl_y2), (255, 255, 0), 2)
        cv2.putText(vis_frame, "Traffic Light", (tl_x1, tl_y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if stop_line is not None:
            x1, y1, x2, y2 = stop_line
            
            if "primary" in detection_method:
                color = (0, 255, 0)  # Green for primary detection
            elif "fallback" in detection_method:
                color = (0, 165, 255)  # Orange for fallback
            else:
                color = (0, 0, 255)  # Red for unknown
            
            cv2.line(vis_frame, (x1, y1), (x2, y2), color, 4)
            
            # Show if stop line is ESTABLISHED or being recalculated
            status = "ESTABLISHED" if self.stop_line_established else "CALCULATING"
            label = f"Stop Line - {status} ({detection_method})"
            cv2.putText(vis_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # **ENHANCED: Show pole base with status**
        if self.fixed_pole_base is not None:
            pole_color = (255, 0, 255) if self.pole_base_established else (128, 0, 128)
            cv2.circle(vis_frame, self.fixed_pole_base, 8, pole_color, -1)
            
            pole_status = "ESTABLISHED" if self.pole_base_established else f"DETECTING ({self.pole_detection_attempts}/{self.max_pole_detection_attempts})"
            cv2.putText(vis_frame, f"Pole Base - {pole_status}", 
                       (self.fixed_pole_base[0] + 10, self.fixed_pole_base[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, pole_color, 2)
        
        # Show pole candidates if still detecting
        if not self.pole_base_established and self.pole_candidates:
            for i, candidate in enumerate(self.pole_candidates):
                cv2.circle(vis_frame, candidate, 4, (200, 100, 200), -1)
                cv2.putText(vis_frame, f"C{i+1}", (candidate[0] + 5, candidate[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 100, 200), 1)
        
        return vis_frame

    def get_pole_base_status(self) -> Dict[str, Any]:
        """Get current pole base status for debugging"""
        return {
            'established': self.pole_base_established,
            'fixed_pole_base': self.fixed_pole_base,
            'detection_attempts': self.pole_detection_attempts,
            'candidates_count': len(self.pole_candidates),
            'candidates': self.pole_candidates
        }

    def save_detection_results(self, stop_lines: List[Dict[str, Any]], frame_shape: Tuple[int, int]):
        """Save detection results in JSON and CSV formats"""
        try:
            detection_data = {
                'frame_shape': frame_shape,
                'timestamp': pd.Timestamp.now().isoformat(),
                'stop_lines': stop_lines
            }
            
            json_path = Path("detection_results.json")
            with open(json_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
            
            if stop_lines:
                csv_data = []
                for sl in stop_lines:
                    row = {
                        'id': sl['id'],
                        'confidence': sl['confidence'],
                        'bbox_x1': sl['bbox'][0],
                        'bbox_y1': sl['bbox'][1],
                        'bbox_x2': sl['bbox'][2],
                        'bbox_y2': sl['bbox'][3],
                        'center_y': sl['center_y'],
                        'polygon_coords': str(sl['polygon'])
                    }
                    csv_data.append(row)
                
                df = pd.DataFrame(csv_data)
                df.to_csv("detection_results.csv", index=False)
                
        except Exception as e:
            print(f"Failed to save detection results: {e}")
