from __future__ import annotations
from typing import List
import os
import shutil

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

# Import your lane detection service and video utility
from app.services.road_mark_detection.road_mark_detection import LaneDetectionService
from app.utils.video_converter import convert_to_browser_compatible

router = APIRouter(prefix="/roadmarks", tags=["Roadmarks"])

# Single service instance (reuse model)
_service = LaneDetectionService()

@router.post("/run")
async def run_roadmark_detection(file: UploadFile = File(...)):
    # Save to input_data directory
    input_dir = "input_data"
    os.makedirs(input_dir, exist_ok=True)
    video_path = os.path.join(input_dir, file.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run detection in background thread
    result = await run_in_threadpool(_service.process_video, video_path)

    # Convert the annotated video (if generated) for browser compatibility
    # The output video path is constructed by the service; we expect it in result["output_video"]
    if result.get("output_video"):
        convert_to_browser_compatible(result["output_video"])

    response = {
        "annotated_video_url": f"http://127.0.0.1:8000/static/{os.path.basename(result['output_video'])}" if result.get("output_video") else None,
        "raw_csv": result.get("raw_csv"),
        "raw_json": result.get("raw_json"),
        "filtered_csv": result.get("filtered_csv"),
        "filtered_json": result.get("filtered_json"),
        "raw_counts": result.get("raw_counts"),
        "filtered_counts": result.get("filtered_counts")
    }
    return response

@router.get("/frames/image")
async def get_frame_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")
