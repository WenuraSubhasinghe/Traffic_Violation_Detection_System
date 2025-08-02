from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
import os
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from app.services.light_violation_detection.light_violation_detection import run_light_violation_detection

router = APIRouter(prefix="/light-violation", tags=["LightViolation"])

load_dotenv()
STATIC_BASE_URL = os.getenv("STATIC_BASE_URL")

# Create a thread pool executor for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

@router.post("/run")
async def run_light_violation(file: UploadFile = File(...)):
    os.makedirs("inputs", exist_ok=True)
    video_path = os.path.join("inputs", file.filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_light_violation_detection, video_path)
        
        annotated_output_path = result.get("output_path")
        annotated_video_url = None
        if annotated_output_path:
            annotated_video_url = f"{STATIC_BASE_URL}{os.path.basename(annotated_output_path)}"
            result["output_path"] = annotated_video_url
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    return result

@router.get("/output-video")
async def get_output_video(path: str):
    # Serve the processed output video file
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")