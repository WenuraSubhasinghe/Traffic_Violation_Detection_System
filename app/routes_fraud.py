from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Dict, Optional, List

import numpy as np
import cv2
import tensorflow as tf
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

from app.services.fraud_detection.fraud_detection import FraudDetectionService

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])
_service = FraudDetectionService()

@router.post("/run")
async def run_fraud_detection(file: UploadFile = File(...)):
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Save the uploaded file
    file_path = os.path.join("outputs", file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run fraud detection (demo, hardcoded sequences)
    result = await _service.run_fraud_detection(file_path)

    return result
