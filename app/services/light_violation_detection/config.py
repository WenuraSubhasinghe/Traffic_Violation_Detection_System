from yaml import safe_load
from pathlib import Path

with open('./config.yaml', 'r') as yaml_file: 
    CONFIG = safe_load(yaml_file)

LIVE = CONFIG['live_mode']

# Enhanced model configuration
MODELS = CONFIG['models']
DETECTION_SETTINGS = CONFIG['detection_settings']
CONFIDENCE_THRESHOLDS = CONFIG['confidence_thresholds']

# Legacy support
MODEL = MODELS['default']

COLORS = {k: tuple(v[::-1]) for k, v in CONFIG['color_pallete'].items()}
CLASSES = CONFIG['object_classes']

PALLETES = {
    'red'    : COLORS['red'],
    'yellow' : COLORS['amber'],
    'green'  : COLORS['lime'],
    "light's off"  : COLORS['emeralad'],
    'car'       : COLORS['cyan'],
    'motorcycle': COLORS['blue'],
    'bus'       : COLORS['violet'],
    'truck'     : COLORS['fucshia'],
    'traffic light': COLORS['rose']
}

def get_model_path(model_type: str) -> str:
    """Get model path based on type"""
    return MODELS.get(model_type, MODELS['default'])
