from __future__ import annotations
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from fastapi.concurrency import run_in_threadpool
from fastapi import UploadFile


class FraudDetectionService:
    """Service for detecting fraudulent driving behavior using a trained LSTM model."""

    def __init__(self, model_path: str = "models/fraud_detection_models/fraud_detection_lstm.h5", output_dir: str = "outputs"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model = tf.keras.models.load_model(model_path)

    def _get_hardcoded_sequences(self):
        vehicle1_sequence = np.array([
            [60, 0, 0, 0, 0],
            [65, 1, 0, 0, 0],
            [70, 1, 0, 0, 1],
            [80, 1, 0, 1, 1],
            [75, 1, 0, 0, 0],
            [60, 1, 0, 0, 0],
            [55, 0, 0, 0, 0],
            [50, 0, 0, 0, 0],
            [45, 0, 0, 0, 0],
            [40, 0, 0, 0, 0],
        ])
        vehicle2_sequence = np.array([
            [45, 0, 0, 0, 0],
            [42, 0, 0, 0, 0],
            [40, 0, 0, 0, 0],
            [38, 0, 0, 0, 0],
            [36, 0, 0, 0, 0],
            [35, 0, 0, 0, 0],
            [34, 0, 0, 0, 0],
            [33, 0, 0, 0, 0],
            [32, 0, 0, 0, 0],
            [30, 0, 0, 0, 0],
        ])
        return vehicle1_sequence, vehicle2_sequence

    def run_fraud_detection_sync(self, video_path: str) -> Dict[str, Any]:
        # Predict fraud probabilities using hardcoded sequences
        v1_seq, v2_seq = self._get_hardcoded_sequences()

        v1_input = v1_seq.reshape((1, 10, 5))
        v2_input = v2_seq.reshape((1, 10, 5))

        fraud_prob_v1 = float(self.model.predict(v1_input)[0][0])
        fraud_prob_v2 = float(self.model.predict(v2_input)[0][0])

        most_fraudulent = "vehicle_1" if fraud_prob_v1 > fraud_prob_v2 else "vehicle_2"

        return {
            "video_path": video_path,
            "vehicle_1_fraud_probability": fraud_prob_v1,
            "vehicle_2_fraud_probability": fraud_prob_v2,
            "most_fraudulent_vehicle": most_fraudulent,
            "detected_accident": "outputs/annotated_test3.jpg"
        }

    async def run_fraud_detection(self, video_path: str) -> Dict[str, Any]:
        return await run_in_threadpool(self.run_fraud_detection_sync, video_path)


# Optional shared instance if needed across modules
_service_cache: Optional[FraudDetectionService] = None

async def run_fraud_detection_async(video_path: str) -> Dict[str, Any]:
    global _service_cache
    if _service_cache is None:
        _service_cache = FraudDetectionService()
    return await _service_cache.run_fraud_detection(video_path)
