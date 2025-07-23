from pydantic import BaseModel
from typing import List

class AccidentConfig(BaseModel):
    # Model paths
    vehicle_model_path: str = "models/yolov8s.pt"   # yolov8s.pt or similar
    accident_model_path: str = "models/accident_detection_model.pt"  # custom accident classifier/detector

    # Vehicle Detection
    vehicle_confidence_threshold: float = 0.5
    vehicle_classes: List[int] = [2, 3, 5, 7]  # car, motorcycle, bus, truck

    # Tracking
    tracker_max_age: int = 30
    tracker_n_init: int = 3

    # Collision (AABB)
    collision_threshold_percent: float = 0.05  # 5%
    collision_padding: int = 20                # px expansion when cropping region for accident model

    # Accident Confirmation
    accident_confidence_threshold: float = 0.7
    accident_confidence_threshold_realtime: float = 0.6

    # Output
    save_accident_frames: bool = True
    accident_frames_dir: str = "outputs/accident_frames"
    save_annotated_video: bool = True

    # Progress Display / Performance (not used directly in service yet)
    display_progress_interval: int = 30
    realtime_display_interval: int = 10
    batch_processing: bool = True
    gpu_acceleration: bool = True