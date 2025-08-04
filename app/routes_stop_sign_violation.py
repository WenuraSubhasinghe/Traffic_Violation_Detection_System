from __future__ import annotations
from typing import List
import os
import shutil

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

# Import your stop sign violation detection service and video utility
from app.services.stop_sign_violation.stop_sign_violation import ViolationDetectionService
from app.utils.video_converter import convert_to_browser_compatible

router = APIRouter(prefix="/stop-sign-violations", tags=["StopSignViolations"])

# Single service instance (reuse model)
_service = ViolationDetectionService()

@router.post("/run")
async def run_stop_sign_violation_detection(file: UploadFile = File(...)):
    # Save to input_data directory
    input_dir = "input_data"
    os.makedirs(input_dir, exist_ok=True)
    video_path = os.path.join(input_dir, file.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run detection in background thread
    result = await run_in_threadpool(_service.process_video, video_path)

    # Convert the annotated video (if generated) for browser compatibility
    if result.get("output_video"):
        convert_to_browser_compatible(result["output_video"])

    response = {
        "annotated_video_url": f"http://127.0.0.1:8000/static/{os.path.basename(result['output_video'])}" if result.get("output_video") else None,
        "violation_images_dir": result.get("violations_dir"),
        "violation_json": result.get("violation_json"),
        "violating_vehicle_ids": result.get("violating_vehicle_ids"),
    }
    return response

@router.get("/frames/image")
async def get_frame_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")
