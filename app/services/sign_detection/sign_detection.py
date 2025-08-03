from ultralytics import YOLO
import cv2
import os
import pandas as pd
import json
from collections import Counter

class SignDetectionService:
    def __init__(self, model_path: str = "models/sign_detection_model/best_roadsigns2.pt", output_dir: str = "outputs"):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_video(self, video_path: str) -> dict:
        results = self.model.predict(
            source=video_path,
            save=True,
            save_txt=True,
            stream=True
        )
        
        class_counts = Counter()
        last_frame_result = None
        for frame_result in results:
            class_ids = frame_result.boxes.cls.tolist()
            class_counts.update(class_ids)
            last_frame_result = frame_result
        
        summary_data = []
        for class_id, count in class_counts.items():
            class_label = self.model.names.get(int(class_id), f"Unknown (ID {int(class_id)})")
            summary_data.append({
                "class_id": int(class_id),
                "class_name": class_label,
                "count": count
            })
        
        out_dir = last_frame_result.save_dir if last_frame_result else self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        
        output_video_path = os.path.join(out_dir, os.path.basename(video_path))
        
        csv_path = os.path.join(out_dir, "video_prediction_summary.csv")
        pd.DataFrame(summary_data).to_csv(csv_path, index=False)
        
        json_path = os.path.join(out_dir, "video_prediction_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=4)
        
        return {
            "output_video": output_video_path,
            "csv_summary": csv_path,
            "json_summary": json_path,
            "class_counts": summary_data
        }
