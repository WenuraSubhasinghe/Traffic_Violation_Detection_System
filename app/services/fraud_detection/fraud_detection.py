from __future__ import annotations
import os
import numpy as np
import tensorflow as tf
import joblib
from typing import Any, Dict, List
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator

class VehicleData(BaseModel):
    speeds: List[float] = Field(..., min_items=10, max_items=10)
    speed_violations: List[int] = Field(..., min_items=10, max_items=10)
    red_light_violations: List[int] = Field(..., min_items=10, max_items=10)
    lane_violations: List[int] = Field(..., min_items=10, max_items=10)

    @validator('speed_violations', 'red_light_violations', 'lane_violations')
    def check_binary(cls, v):
        if not all(x in [0, 1] for x in v):
            raise ValueError('Violation fields must contain only 0 or 1')
        return v

class FraudDetectionInput(BaseModel):
    vehicles: List[VehicleData] = Field(..., min_items=1, max_items=4)
    area_type: str
    speed_limit: float = Field(..., gt=0)

    @validator('area_type')
    def check_area_type(cls, v):
        valid_areas = ['urban', 'highway', 'residential', 'expressway']
        if v not in valid_areas:
            raise ValueError(f'area_type must be one of {valid_areas}')
        return v

class FraudDetectionService:
    """Service for detecting fraudulent driving behavior using a trained LSTM model."""

    def __init__(
        self,
        model_path: str = "models/fraud_detection_models/fraud_detection_lstm.h5",
        scaler_path: str = "models/fraud_detection_models/speed_scaler.pkl",
        output_dir: str = "outputs"
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.output_dir = output_dir
        self.time_steps = 10
        self.num_features = 4
        self.max_vehicles = 4
        os.makedirs(output_dir, exist_ok=True)
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")

    def _preprocess_input(self, input_data: FraudDetectionInput) -> tuple[np.ndarray, np.ndarray, int]:
        """Preprocess vehicle data to match model input requirements."""
        vehicles_data = input_data.vehicles
        area_type = input_data.area_type
        speed_limit = input_data.speed_limit
        num_real_vehicles = len(vehicles_data)

        # Prepare static features
        area_types = ['highway', 'residential', 'expressway']
        static_features = [speed_limit]
        for area in area_types:
            static_features.append(1.0 if area_type == area else 0.0)

        # Initialize arrays
        time_series_data = []
        static_data = []

        # Process each vehicle
        for vehicle in vehicles_data:
            vehicle_ts = []
            for t in range(self.time_steps):
                timestep_data = [
                    vehicle.speeds[t],
                    vehicle.speed_violations[t],
                    vehicle.red_light_violations[t],
                    vehicle.lane_violations[t]
                ]
                vehicle_ts.append(timestep_data)
            time_series_data.append(vehicle_ts)
            static_data.append(static_features.copy())

        # Pad to max_vehicles if needed
        while len(time_series_data) < self.max_vehicles:
            dummy_ts = [[0.0] * self.num_features] * self.time_steps
            dummy_static = [0.0] * len(static_features)
            time_series_data.append(dummy_ts)
            static_data.append(dummy_static)

        # Convert to numpy arrays and add batch dimension
        X_time_series = np.array([time_series_data])  # Shape: (1, max_vehicles, time_steps, num_features)
        X_static = np.array([static_data])            # Shape: (1, max_vehicles, num_static_features)

        # Normalize speed data (feature 0)
        original_shape = X_time_series.shape
        X_ts_reshaped = X_time_series.reshape(-1, self.num_features)
        X_ts_normalized = X_ts_reshaped.copy().astype(float)
        X_ts_normalized[:, 0] = self.scaler.transform(X_ts_reshaped[:, 0].reshape(-1, 1)).flatten()
        X_time_series = X_ts_normalized.reshape(original_shape)

        return X_time_series, X_static, num_real_vehicles

    def run_fraud_detection_sync(self, input_data: FraudDetectionInput) -> Dict[str, Any]:
        """Run fraud detection on the provided vehicle data."""
        try:
            # Preprocess input
            X_time_series, X_static, num_real_vehicles = self._preprocess_input(input_data)

            # Make prediction
            predictions = self.model.predict([X_time_series, X_static], verbose=0)
            predictions = predictions[0, :num_real_vehicles, 0]  # Shape: (num_real_vehicles,)

            # Prepare results
            result = {
                "area_type": input_data.area_type,
                "speed_limit": input_data.speed_limit,
                "num_vehicles": num_real_vehicles,
                "vehicle_predictions": []
            }

            most_fraudulent_idx = np.argmax(predictions) if len(predictions) > 0 else -1
            for i in range(num_real_vehicles):
                fraud_prob = float(predictions[i])
                is_fraud = fraud_prob > 0.5
                confidence = fraud_prob if is_fraud else (1 - fraud_prob)
                vehicle_result = {
                    "vehicle_id": i + 1,
                    "fraud_probability": fraud_prob,
                    "is_fraudulent": is_fraud,
                    "confidence": confidence,
                    "avg_speed": float(np.mean(input_data.vehicles[i].speeds)),
                    "total_violations": int(
                        sum(input_data.vehicles[i].speed_violations) +
                        sum(input_data.vehicles[i].red_light_violations) +
                        sum(input_data.vehicles[i].lane_violations)
                    )
                }
                result["vehicle_predictions"].append(vehicle_result)

            result["most_fraudulent_vehicle"] = f"vehicle_{most_fraudulent_idx + 1}" if most_fraudulent_idx >= 0 else None

            return result

        except Exception as e:
            raise ValueError(f"Error during fraud detection: {str(e)}")

    async def run_fraud_detection(self, input_data: FraudDetectionInput) -> Dict[str, Any]:
        """Async wrapper for fraud detection."""
        return await run_in_threadpool(self.run_fraud_detection_sync, input_data)