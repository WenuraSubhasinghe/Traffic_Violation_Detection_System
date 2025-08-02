import cv2
import numpy as np
from typing import Tuple, List, Optional

from app.services.light_violation_detection.stop_line_detector import EnhancedStopLineDetector

class EnhancedViolationDetector:
    def __init__(self):
        self.stop_line_detector = EnhancedStopLineDetector()
        self.fallback_offset = 180
        self.frame_confirmations = {} 
        
    def establish_violation_boundary(self, frame: np.ndarray, 
                                   traffic_light_bbox: Tuple[int, int, int, int],
                                   light_state: str) -> Tuple[Tuple[int, int, int, int], str]:
        """Establish violation boundary using the same logic as stop_line_detector.py"""
        
        # Use the persistent stop line detection method from enhanced_stop_line_detector
        detected_stop_line, detection_method = self.stop_line_detector.establish_persistent_stop_line(
            frame, traffic_light_bbox, light_state
        )
        
        if detected_stop_line:
            return detected_stop_line, detection_method
        else:
            # Fallback to traffic light-based boundary
            fallback_line = self.create_fallback_boundary(frame, traffic_light_bbox)
            return fallback_line, "FALLBACK_BOUNDARY"
    
    def reset_stop_line_cache(self):
        """Reset the stop line cache - delegates to stop_line_detector"""
        self.stop_line_detector.reset_for_new_light_cycle()
    
    def create_fallback_boundary(self, frame: np.ndarray, 
                               traffic_light_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Create fallback violation boundary based on traffic light position"""
        tl_x1, tl_y1, tl_x2, tl_y2 = traffic_light_bbox
        frame_height, frame_width = frame.shape[:2]
        
        boundary_y = min(frame_height - 10, tl_y2 + self.fallback_offset)
        line_start_x = int(frame_width * 0.15)
        line_end_x = int(frame_width * 0.85)
        
        return (line_start_x, boundary_y, line_end_x, boundary_y)

    def check_vehicle_violation(self,
                              vehicle_bbox: Tuple[int, int, int, int],
                              violation_boundary: Tuple[int, int, int, int],
                              vehicle_id: int,
                              previous_positions: dict,
                              frame_confirmations: dict,
                              required_frames: int = 3) -> bool:
        """Improved violation check with multiple bottom points"""
        
        veh_x1, veh_y1, veh_x2, veh_y2 = vehicle_bbox
        line_x1, line_y1, line_x2, line_y2 = violation_boundary

        # Define bottom points of the vehicle (left, center, right)
        vehicle_bottom_points = [
            (veh_x1, veh_y2),                       # bottom-left
            ((veh_x1 + veh_x2) // 2, veh_y2),       # bottom-center
            (veh_x2, veh_y2)                        # bottom-right
        ]

        # Create stop line polygon area (a thin rectangle around the line)
        line_thickness = 15
        stop_line_polygon = np.array([
            (line_x1, line_y1 - line_thickness),
            (line_x2, line_y2 - line_thickness),
            (line_x2, line_y2 + line_thickness),
            (line_x1, line_y1 + line_thickness)
        ], dtype=np.int32)

        # Check if any bottom point is past the line in terms of Y-coordinate
        crossed_line_y = False
        for (px, py) in vehicle_bottom_points:
            if line_x2 != line_x1:
                t = (px - line_x1) / (line_x2 - line_x1)
                violation_line_y = line_y1 + t * (line_y2 - line_y1)
            else:
                violation_line_y = line_y1

            if py > violation_line_y:
                crossed_line_y = True
                break

        # Check polygon intersection with vehicle bottom edge
        crossed_polygon = any(
            cv2.pointPolygonTest(stop_line_polygon, (float(px), float(py)), False) >= 0
            for (px, py) in vehicle_bottom_points
        )

        # Combine checks
        crossed = crossed_line_y or crossed_polygon

        # Check history to avoid false positives
        if vehicle_id not in previous_positions:
            previous_positions[vehicle_id] = veh_y2
            # If first seen already past the line, never mark violation
            violation_line_y = (line_y1 + line_y2) // 2
            if veh_y2 > violation_line_y:
                return False

        # Existing check
        violation_line_y = (line_y1 + line_y2) // 2
        was_behind_line = previous_positions[vehicle_id] <= violation_line_y
        violation_detected = crossed and was_behind_line

        # Update vehicle position history
        previous_positions[vehicle_id] = veh_y2

        # Multi-frame confirmation
        if violation_detected:
            frame_confirmations[vehicle_id] = frame_confirmations.get(vehicle_id, 0) + 1
        else:
            frame_confirmations[vehicle_id] = 0

        return frame_confirmations[vehicle_id] >= required_frames
    
    def visualize_violation_detection(self, frame: np.ndarray, 
                                    traffic_light_bbox: Tuple[int, int, int, int],
                                    violation_boundary: Tuple[int, int, int, int],
                                    detection_method: str,
                                    violating_vehicles: List = None) -> np.ndarray:
        """Enhanced visualization with detection method info"""
        vis_frame = frame.copy()
        
        # Draw traffic light
        tl_x1, tl_y1, tl_x2, tl_y2 = traffic_light_bbox
        cv2.rectangle(vis_frame, (tl_x1, tl_y1), (tl_x2, tl_y2), (255, 255, 0), 2)
        
        # Draw violation boundary with different colors based on method
        line_x1, line_y1, line_x2, line_y2 = violation_boundary
        
        # Updated to match the detection methods from stop_line_detector.py
        if detection_method == "primary_multiple_validated":
            line_color = (0, 255, 0)  # Green for primary multiple validated
            method_text = "PRIMARY MULTIPLE VALIDATED"
        elif detection_method == "primary_single_validated":
            line_color = (0, 255, 255)  # Yellow for primary single validated
            method_text = "PRIMARY SINGLE VALIDATED"
        elif detection_method == "fallback_offset":
            line_color = (0, 165, 255)  # Orange for fallback offset
            method_text = "FALLBACK OFFSET"
        elif detection_method == "FALLBACK_BOUNDARY":
            line_color = (255, 0, 0)  # Blue for fallback boundary
            method_text = "FALLBACK BOUNDARY"
        else:
            line_color = (128, 128, 128)  # Gray for unknown method
            method_text = f"UNKNOWN: {detection_method}"
        
        cv2.line(vis_frame, (line_x1, line_y1), (line_x2, line_y2), line_color, 4)
        cv2.putText(vis_frame, method_text, (line_x1, line_y1 - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        
        # Draw violating vehicles
        if violating_vehicles:
            for vehicle in violating_vehicles:
                veh_x1, veh_y1, veh_x2, veh_y2 = vehicle[:4]
                cv2.rectangle(vis_frame, (int(veh_x1), int(veh_y1)), 
                            (int(veh_x2), int(veh_y2)), (0, 0, 255), 3)
                cv2.putText(vis_frame, "VIOLATION", (int(veh_x1), int(veh_y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return vis_frame
