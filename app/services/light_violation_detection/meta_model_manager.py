from ultralytics import YOLO
import numpy as np
import os
import logging

from app.services.light_violation_detection.condition_detector import ConditionDetector
from app.services.light_violation_detection.config import CONFIG

# Setup logging (do this once in your main entrypoint or __main__)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s | %(levelname)s | %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class MetaModelManager:
    def __init__(self):
        self.condition_detector = ConditionDetector()
        self.meta_model = None
        self.default_model = None
        self.use_meta_yolo = CONFIG['detection_settings']['use_meta_yolo']
        self.last_model_type = None   # Store last-used model type (for switch detection)
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load available models"""
        meta_path = CONFIG['models']['meta_yolo']
        default_path = CONFIG['models']['default']
        
    
        if os.path.exists(meta_path) and self.use_meta_yolo:
            logger.info(f"Loading Meta-YOLOv8 from {meta_path}")
            self.meta_model = YOLO(model=meta_path)
        else:
            logger.info(f"Meta-YOLOv8 not found at {meta_path}, using default model")
            
        # Load default model as fallback

        logger.info(f"Loading default model from {default_path}")
        self.default_model = YOLO(model=default_path)
    
    def get_optimal_model_and_settings(self, frame: np.ndarray) -> tuple:
        conditions = self.condition_detector.analyze_frame_conditions(frame)

        if self.meta_model is not None and conditions['is_challenging']:
            model = self.meta_model
            model_type = "meta_yolo"
        else:
            model = self.default_model
            model_type = "default"

        # If model type changed, log the switch event
        if model_type != self.last_model_type:
            logger.info(
                f"Model switch: {self.last_model_type or 'None'} → {model_type} "
                f"(Reason: {'Challenging' if conditions['is_challenging'] else 'Clear'} condition: {conditions['condition_type']})"
            )
            self.last_model_type = model_type  # Update last_model_type

        confidence = self.condition_detector.get_optimal_confidence(conditions)
        return model, model_type, confidence, conditions
