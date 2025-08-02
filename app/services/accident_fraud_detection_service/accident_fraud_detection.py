from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

from app.services.accident_detection.accident_detection import AccidentDetectionService
from app.services.fraud_detection.fraud_detection import (
    FraudDetectionService, FraudDetectionInput, VehicleData
)

class AccidentAndFraudDetectionService:
    def __init__(self):
        self.accident_service = AccidentDetectionService()
        self.fraud_service = FraudDetectionService()

    async def process_video(self, video_path: str) -> Dict[str, Any]:
        # Run accident detection
        accident_result = await self.accident_service.process_video(video_path)

        # Default: no accident, no fraud check
        accident_frames = accident_result.get("accident_frames", [])
        if not accident_frames:
            return {
                "accident_details": accident_result,
                "fraudulence_details": None,
                "message": "No accident detected."
            }

        # Identify involved vehicles (collect unique track_ids from all accident frames)
        involved_vehicle_ids = set()
        for frame in accident_frames:
            for vid in frame.get("track_ids", []):
                involved_vehicle_ids.add(vid)
        involved_vehicle_ids = sorted(list(involved_vehicle_ids))

        # --- MOCK: Retrieve violation details for each vehicle ---
        # In real use, fetch from DB. Here, generate mock data.
        vehicles_data = []
        for vid in involved_vehicle_ids:
            # Mock: random speeds and violations for 10 time steps
            speeds = list(np.random.uniform(20, 80, size=10))
            speed_violations = list(np.random.choice([0, 1], size=10))
            red_light_violations = list(np.random.choice([0, 1], size=10))
            lane_violations = list(np.random.choice([0, 1], size=10))
            vehicles_data.append(VehicleData(
                speeds=speeds,
                speed_violations=speed_violations,
                red_light_violations=red_light_violations,
                lane_violations=lane_violations
            ))

        # Use area_type and speed_limit from accident_result if available, else mock
        area_type = accident_result.get("area_type", "urban")
        speed_limit = accident_result.get("speed_limit", 50.0)

        fraud_input = FraudDetectionInput(
            vehicles=vehicles_data,
            area_type=area_type,
            speed_limit=speed_limit
        )

        # Run fraud detection
        fraud_result = await self.fraud_service.run_fraud_detection(fraud_input)

        return {
            "accident_details": accident_result,
            "fraudulence_details": fraud_result,
            "message": "Accident detected and fraudulence analyzed."
        }