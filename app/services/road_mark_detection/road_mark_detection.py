from ultralytics import YOLO
import os
import pandas as pd
import json
from collections import Counter

class LaneDetectionService:
    def __init__(
        self,
        model_path: str = "models\road_mark_detection_model\road_mark_best.pt",
        output_dir: str = "outputs"
    ):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Define class-specific confidence thresholds here if needed
        self.class_conf_thresholds = {
            0: 0.60,  # Center_Dash
            1: 0.90,  # Center_Solid
            2: 0.30,  # LA
            3: 0.50,  # Lane_Boundary
            4: 0.30,  # RA
            5: 0.50,  # SA
            6: 0.30,  # SLA
            7: 0.20,  # SRA
            8: 0.60   # Stop_Line
        }

    def process_video(self, video_path: str) -> dict:
        results = self.model.predict(
            source=video_path,
            save=True,
            save_txt=True,
            stream=True,
            conf=0.1  # Set global conf low to allow class-specific filtering
        )

        raw_counts = Counter()
        filtered_counts = Counter()
        last_frame_result = None

        for frame_result in results:
            boxes = frame_result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes.cls)):
                class_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                raw_counts[class_id] += 1
                threshold = self.class_conf_thresholds.get(class_id, 0.5)
                if conf >= threshold:
                    filtered_counts[class_id] += 1
            last_frame_result = frame_result

        class_names = self.model.names

        def convert_summary(counter_dict):
            summary = []
            for class_id, count in counter_dict.items():
                summary.append({
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"Unknown (ID {class_id})"),
                    "count": count
                })
            return summary

        raw_summary = convert_summary(raw_counts)
        filtered_summary = convert_summary(filtered_counts)

        out_dir = last_frame_result.save_dir if last_frame_result else self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        output_video_path = os.path.join(out_dir, os.path.basename(video_path))

        # Save summaries
        raw_csv_path = os.path.join(out_dir, "video_prediction_summary_raw.csv")
        raw_json_path = os.path.join(out_dir, "video_prediction_summary_raw.json")
        pd.DataFrame(raw_summary).to_csv(raw_csv_path, index=False)
        with open(raw_json_path, "w") as f:
            json.dump(raw_summary, f, indent=4)

        filtered_csv_path = os.path.join(out_dir, "video_prediction_summary_filtered.csv")
        filtered_json_path = os.path.join(out_dir, "video_prediction_summary_filtered.json")
        pd.DataFrame(filtered_summary).to_csv(filtered_csv_path, index=False)
        with open(filtered_json_path, "w") as f:
            json.dump(filtered_summary, f, indent=4)

        return {
            "output_video": output_video_path,
            "raw_csv": raw_csv_path,
            "raw_json": raw_json_path,
            "filtered_csv": filtered_csv_path,
            "filtered_json": filtered_json_path,
            "raw_counts": raw_summary,
            "filtered_counts": filtered_summary
        }
