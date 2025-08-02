from __future__ import annotations
import os
import shutil
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse

from app.services.accident_fraud_detection_service import AccidentAndFraudDetectionService

router = APIRouter(prefix="/accident-fraud", tags=["Accident & Fraud Detection"])
_service = AccidentAndFraudDetectionService()

@router.post("/run")
async def run_accident_and_fraud_detection(file: UploadFile = File(...)):
    os.makedirs("inputs", exist_ok=True)
    video_path = os.path.join("inputs", file.filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        result = await _service.process_video(video_path)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")