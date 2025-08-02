from numpy import array, ndarray
from cv2 import inRange, countNonZero
import cv2

def maskColor(frame_area: ndarray, validateYellow: bool = False):
    """Original proven color masking logic"""
    # Red color ranges (two ranges to handle hue wraparound)
    maskRed = inRange(frame_area, array([0, 100, 100]), array([10, 255, 255]))
    maskRed += inRange(frame_area, array([160, 100, 100]), array([179, 255, 255]))
    
    if validateYellow:
        # For yellow validation, only check yellow range
        maskYellow = inRange(frame_area, array([20, 100, 100]), array([30, 255, 255]))
        yellow = countNonZero(maskYellow)
        return yellow
    else:
        # Yellow and green ranges
        maskYellow = inRange(frame_area, array([20, 100, 100]), array([30, 255, 255]))
        maskGreen = inRange(frame_area, array([40, 100, 100]), array([80, 255, 255]))
    
    red = countNonZero(maskRed)
    yellow = countNonZero(maskYellow)
    green = countNonZero(maskGreen)
    
    return (red, yellow, green)

def position_based_color_analysis(frame_area: ndarray):
    """Analyze colors based on their expected positions in traffic light"""
    height, width = frame_area.shape[:2]
    
    if height < 30 or width < 20:  # Skip too small detections
        return {'valid': False}
    
    # Divide traffic light into 3 vertical sections based on standard layout
    section_height = height // 3
    top_section = frame_area[0:section_height, :]                    # Red zone (top)
    middle_section = frame_area[section_height:2*section_height, :]  # Yellow zone (middle)
    bottom_section = frame_area[2*section_height:height, :]          # Green zone (bottom)
    
    # Count colors in their expected positions
    # Red should be in TOP section
    red_top = countNonZero(inRange(top_section, array([0, 100, 100]), array([10, 255, 255])))
    red_top += countNonZero(inRange(top_section, array([160, 100, 100]), array([179, 255, 255])))
    
    # Yellow should be in MIDDLE section
    yellow_middle = countNonZero(inRange(middle_section, array([20, 100, 100]), array([30, 255, 255])))
    
    # Green should be in BOTTOM section
    green_bottom = countNonZero(inRange(bottom_section, array([40, 100, 100]), array([80, 255, 255])))
    
    # Check for color bleeding (misclassification indicators)
    red_middle = countNonZero(inRange(middle_section, array([0, 100, 100]), array([10, 255, 255])))
    red_middle += countNonZero(inRange(middle_section, array([160, 100, 100]), array([179, 255, 255])))
    
    red_bottom = countNonZero(inRange(bottom_section, array([0, 100, 100]), array([10, 255, 255])))
    red_bottom += countNonZero(inRange(bottom_section, array([160, 100, 100]), array([179, 255, 255])))
    
    return {
        'valid': True,
        'red_top': red_top,
        'yellow_middle': yellow_middle,
        'green_bottom': green_bottom,
        'red_middle': red_middle,
        'red_bottom': red_bottom,
        'height': height,
        'width': width
    }
    
def estimate_colors_from_pixels(frame, traffic_lights):
    """Estimate the color (red, yellow, green) for each detected traffic light.
    Args:
        frame: Input image, assumed to be BGR (as from OpenCV).
        traffic_lights: List/array of bounding boxes, format [x1, y1, x2, y2] per light.
    Returns:
        dict: {'red': [(idx, confidence)], 'yellow': [(idx, confidence)], 'green': [(idx, confidence)]}
              where idx is the index in traffic_lights and confidence is a pixel count or similar measure.
    """
    detected_colors = {'red': [], 'yellow': [], 'green': []}
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for idx, bbox in enumerate(traffic_lights):
        # Handle various bbox formats, e.g., torch.Tensor or numpy.
        if hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        x1, y1, x2, y2 = [int(val) for val in bbox[:4]]

        # Extract ROI and check for valid size
        roi = hsv_frame[y1:y2, x1:x2]  # HSV region of interest

        analysis = position_based_color_analysis(roi)
        red_valid = validate_red_position(analysis)
        yellow_valid = validate_yellow_position(analysis)
        green_valid = validate_green_position(analysis)

        # Use sum of color pixels as a confidence measure
        if red_valid:
            confidence = analysis.get('red_top', 0)
            detected_colors['red'].append((idx, confidence))
        elif yellow_valid:
            confidence = analysis.get('yellow_middle', 0)
            detected_colors['yellow'].append((idx, confidence))
        elif green_valid:
            confidence = analysis.get('green_bottom', 0)
            detected_colors['green'].append((idx, confidence))

    return detected_colors

def validate_red_position(analysis: dict) -> bool:
    """Validate that red color is actually in the top section where it should be"""
    if not analysis['valid']:
        return False
    
    red_top = analysis['red_top']
    red_middle = analysis['red_middle']
    red_bottom = analysis['red_bottom']
    
    # Red light validation criteria:
    # 1. Must have significant red pixels in top section
    # 2. Red in top should be much more than middle/bottom
    # 3. Prevent yellow-to-red misclassification
    
    min_red_threshold = 15
    position_ratio_threshold = 2.0  # Top should have 2x more red than middle
    
    if red_top < min_red_threshold:
        return False
    
    # Red should be concentrated in TOP section, not middle
    if red_middle > 0 and red_top < red_middle * position_ratio_threshold:
        return False
    
    # Red should be concentrated in TOP section, not bottom
    if red_bottom > 0 and red_top < red_bottom * position_ratio_threshold:
        return False
    
    return True

def validate_yellow_position(analysis: dict) -> bool:
    """Validate that yellow color is in the middle section where it should be"""
    if not analysis['valid']:
        return False
    
    yellow_middle = analysis['yellow_middle']
    red_middle = analysis['red_middle']
    
    # Yellow light validation criteria:
    # 1. Must have significant yellow pixels in middle section
    # 2. Middle section should not be dominated by red
    # 3. Yellow should be concentrated in middle, not bleeding from top
    
    min_yellow_threshold = 15
    max_red_interference = 10
    
    if yellow_middle < min_yellow_threshold:
        return False
    
    # Ensure middle section isn't dominated by red (which would indicate red light)
    if red_middle > max_red_interference and red_middle > yellow_middle:
        return False
    
    return True

def validate_green_position(analysis: dict) -> bool:
    """Validate that green color is in the bottom section where it should be"""
    if not analysis['valid']:
        return False
    
    green_bottom = analysis['green_bottom']
    min_green_threshold = 15
    
    return green_bottom >= min_green_threshold

def validateYellow(frame_area: ndarray) -> bool:
    """Enhanced yellow validation with position checking"""
    analysis = position_based_color_analysis(frame_area)
    return validate_yellow_position(analysis)

def recognize_color(frame, traffic_lights, print_info=False, model_type=None, model_names=None):
    detected_colors = {'red': [], 'yellow': [], 'green': []}

    if model_type == "meta_yolo" and model_names:
        for i, bbox in enumerate(traffic_lights):
            cls_id = int(bbox[-1].item())
            color_name = model_names.get(cls_id, 'unknown')
            if color_name in detected_colors:
                detected_colors[color_name].append(i)
    else:
        # fallback to color estimation by pixel analysis
        detected_colors = estimate_colors_from_pixels(frame, traffic_lights)
        
    return detected_colors


def chooseOne(light_colors: dict) -> tuple:
    """Choose the most confident traffic light detection with priority"""
    max_confidence = 0
    chosen_index = -1
    chosen_confidence = 0
    
    # Priority: Red > Yellow > Green (for violation detection)
    for color_type in ['red', 'yellow', 'green']:
        if light_colors[color_type]:
            for index, confidence in light_colors[color_type]:
                if confidence > max_confidence:
                    max_confidence = confidence
                    chosen_index = index
                    chosen_confidence = confidence
            if chosen_index >= 0:  # Found a light in this color
                break
    
    return (chosen_index, chosen_confidence)

def debug_traffic_light_sections(frame_area: ndarray, light_index: int = 0):
    """Debug function to visualize traffic light sections and color distribution"""
    analysis = position_based_color_analysis(frame_area)
    
    if not analysis['valid']:
        print(f"Traffic light {light_index}: Invalid size")
        return
    
    print(f"\nTraffic Light {light_index} Analysis:")
    print(f"  Dimensions: {analysis['width']}x{analysis['height']}")
    print(f"  Red in TOP section: {analysis['red_top']}")
    print(f"  Yellow in MIDDLE section: {analysis['yellow_middle']}")
    print(f"  Green in BOTTOM section: {analysis['green_bottom']}")
    print(f"  Red bleeding into MIDDLE: {analysis['red_middle']}")
    print(f"  Red bleeding into BOTTOM: {analysis['red_bottom']}")
    
    print(f"  Position Validation:")
    print(f"    Red valid: {validate_red_position(analysis)}")
    print(f"    Yellow valid: {validate_yellow_position(analysis)}")
    print(f"    Green valid: {validate_green_position(analysis)}")
