import cv2
import numpy as np
from typing import Tuple

class ConditionDetector:
    def __init__(self):
        self.brightness_threshold = 80
        self.contrast_threshold = 30
        
    def analyze_frame_conditions(self, frame: np.ndarray) -> dict:
        """Analyze frame to determine optimal detection settings"""
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate key metrics
        mean_brightness = np.mean(gray)
        contrast = gray.std()
        saturation = np.mean(hsv[:,:,1])
        
        # Determine conditions
        is_low_light = mean_brightness < self.brightness_threshold
        is_low_contrast = contrast < self.contrast_threshold
        is_low_saturation = saturation < 80
        
        # Weather condition detection
        blue_ratio = np.mean(frame[:,:,0]) / (np.mean(frame) + 1e-6)
        is_foggy_rainy = (is_low_contrast and is_low_saturation) or blue_ratio > 0.4

        condition_type = self._get_condition_type(mean_brightness, contrast, saturation, blue_ratio)

        print(f"DEBUG: Brightness={mean_brightness:.1f}, Contrast={contrast:.1f}, Saturation={saturation:.1f}")
        
        return {
                'brightness': mean_brightness,
                'contrast': contrast,
                'saturation': saturation,
                'is_low_light': mean_brightness < self.brightness_threshold,
                'is_challenging': condition_type != "clear",
                'condition_type': condition_type
        }
    
    def _get_condition_type(self, brightness: float, contrast: float, saturation: float, blue_ratio: float) -> str:
        """Determine specific condition type"""
        if brightness < 90:
            return "night"
        elif contrast < 25 and saturation < 70:
            return "foggy"
        elif blue_ratio > 0.4 and contrast < 35:
            return "rainy"
        elif brightness > 180 and contrast < 25:
            return "snowy"
        else:
            return "clear"
    
    def get_optimal_confidence(self, conditions: dict) -> float:
        """Get optimal confidence threshold based on conditions"""
        if conditions['is_challenging']:
            if conditions['condition_type'] == "night":
                return 0.25
            else:
                return 0.3
        return 0.5
