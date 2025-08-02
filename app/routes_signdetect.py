from __future__ import annotations
from typing import List
import os
import shutil

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

# Import your service (be sure this import path matches your structure)
from app.services.sign_detection.sign_detection import SignDetectionService

router = APIRouter(prefix="/signs", tags=["Signs"])

# Single service instance (reuse model)
_service = SignDetectionService()

@router.post("/run")
async def run_sign_detection(file: UploadFile = File(...)):
    # Save to input_data folder
    input_dir = "input_data"
    os.makedirs(input_dir, exist_ok=True)
    video_path = os.path.join(input_dir, file.filename)
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Run detection in background thread
    result = await run_in_threadpool(_service.process_video, video_path)

    # Optionally augment the result with a public static URL if needed
    response = {
        "annotated_video_url": f"http://127.0.0.1:8000/static/{os.path.basename(result['output_video'])}" if result.get("output_video") else None,
        "csv_summary": result.get("csv_summary"),
        "json_summary": result.get("json_summary"),
        "class_counts": result.get("class_counts")
    }
    return response

@router.get("/frames/image")
async def get_frame_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")
