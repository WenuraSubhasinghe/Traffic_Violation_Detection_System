from __future__ import annotations
import os
import shutil

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from bson import ObjectId

from app.database import db
from app.services.lane_change_detection import LaneChangeDetectionService  # Your lane change detection service

router = APIRouter(prefix="/lanechanges", tags=["LaneChanges"])

# Single shared service instance (loads models once)
_service = LaneChangeDetectionService()


@router.post("/run")
async def run_lane_change_detection(video_name: str = "test_video.mp4"):
    video_path = os.path.join("videos", video_name)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found.")
    doc = await _service.process_video(video_path)
    # Convert ObjectId to str for JSON response
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/logs")
async def list_lane_change_logs(limit: int = 20):
    cursor = db.lane_change_logs.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@router.get("/logs/{log_id}")
async def get_lane_change_log(log_id: str):
    if not ObjectId.is_valid(log_id):
        raise HTTPException(status_code=400, detail="Invalid id")
    doc = await db.lane_change_logs.find_one({"_id": ObjectId(log_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    doc["_id"] = str(doc["_id"])
    return doc


@router.get("/logs/{log_id}/frames")
async def get_lane_change_frames(log_id: str):
    if not ObjectId.is_valid(log_id):
        raise HTTPException(status_code=400, detail="Invalid id")
    doc = await db.lane_change_logs.find_one({"_id": ObjectId(log_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")
    frames = doc.get("lane_change_frames", [])
    return frames


@router.get("/frames/image")
async def get_frame_image(path: str):
    # Serve saved lane change frame image
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


@router.post("/test-image")
async def test_single_image(file: UploadFile = File(...)):
    os.makedirs("outputs", exist_ok=True)

    image_path = os.path.join("outputs", file.filename)
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = await _service.test_single_image(image_path)

    annotated_image = result.get("annotated_image")
    annotated_image_url = None
    if annotated_image:
        annotated_image_url = f"http://127.0.0.1:8000/static/{os.path.basename(annotated_image)}"

    return {
        "detections": result.get("detections", []),
        "lane_changes": result.get("lane_changes", []),
        "annotated_image_url": annotated_image_url,
    }
