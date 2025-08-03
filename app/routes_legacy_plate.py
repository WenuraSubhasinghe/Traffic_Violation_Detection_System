from __future__ import annotations
import os
import shutil
import uuid
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from app.services.legacy_plate_detection.legacy_plate_detection import PlateDetectionService

router = APIRouter(prefix="/plates", tags=["Plates"])
_service = PlateDetectionService()

@router.post("/run")
async def run_plate_detection(file: UploadFile = File(...)):
    input_dir = "input_data"
    os.makedirs(input_dir, exist_ok=True)
    image_path = os.path.join(input_dir, file.filename)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Process in a background thread
    result = await run_in_threadpool(_service.process_image, image_path)
    annotated_images = result.get("annotated_images", [])
    annotated_url = None
    if annotated_images:
        rel_path = os.path.relpath(annotated_images[0], "outputs")
        # Use a clear, unique URL prefix
        annotated_url = f"http://127.0.0.1:8000/static/{rel_path}"
    return {"annotated_image_url": annotated_url}

@router.get("/static_image")
async def get_annotated_image(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")
