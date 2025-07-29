import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.u_turn_detection.u_turn_detection import UTurnDetector
from app.database import db
from datetime import datetime

router = APIRouter(prefix="/uturn", tags=["U-Turn Detection"])
_service = UTurnDetector()

@router.post("/run")
async def run_uturn_detection(file: UploadFile = File(...)):
    """
    Upload a video and run U-turn detection.
    """
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = os.path.join("inputs", file.filename)
    output_path = os.path.join("outputs", f"uturn_{file.filename}")

    try:
        # Save uploaded file
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run detection (await if async, else plain call)
        u_turn_events = (
            await _service.process_video(video_path, output_path, show_display=False)
            if hasattr(_service.process_video, '__await__')
            else _service.process_video(video_path, output_path, show_display=False)
        )

        # Parse U-turn event details (standardize for frontend)
        uturn_details = []
        for event in u_turn_events:
            uturn_details.append({
                "track_id": event.get("track_id"),
                "timestamp": event.get("timestamp"),
                "frame_idx": event.get("frame_idx"),
                "angle": event.get("angle"),
                "position": {
                    "x": event["position"][0] if "position" in event else None,
                    "y": event["position"][1] if "position" in event else None,
                },
                "plate": event.get("plate", None),
            })

        # Group vehicle IDs with at least one U-turn
        vehicles_with_uturn = list({event["track_id"] for event in uturn_details})

        # Prepare per-vehicle event breakdown
        vehicles = []
        for v_id in vehicles_with_uturn:
            vehicle_uturns = [e for e in uturn_details if e["track_id"] == v_id]
            vehicles.append({
                "vehicle_id": v_id,
                "uturn_events": vehicle_uturns,
                "total_uturns": len(vehicle_uturns),
                "plates": [e["plate"] for e in vehicle_uturns if e["plate"]],
            })

        # Build summary
        summary = {
            "total_detected_uturns": len(uturn_details),
            "total_tracked_vehicles": len(getattr(_service, "vehicle_tracks", {})),
            "vehicles_with_uturns": len(vehicles_with_uturn),
            "processed_at": datetime.utcnow(),
        }

        # --- Save results to MongoDB ---
        if uturn_details:
            docs = [{
                "track_id": d["track_id"],
                "timestamp": d["timestamp"],
                "frame_idx": d["frame_idx"],
                "angle": d["angle"],
                "position": d["position"],
                "plate": d.get("plate"),
                "video_path": output_path,
                "created_at": datetime.utcnow()
            } for d in uturn_details]
            # await db.uturn_records.insert_many(docs)

        return {
            "annotated_video_url": f"http://127.0.0.1:8000/static/uturn_{file.filename}",
            "summary": summary,
            "vehicles": vehicles,
            "uturn_events": uturn_details,
        }
    except Exception as e:
    # Always provide a "detail" field
        return {"detail": f"Processing failed: {str(e)}"}

@router.get("/output/{filename}")
async def get_uturn_output_video(filename: str):
    """
    Download the annotated video for U-turn detection.
    """
    file_path = os.path.join("outputs", f"uturn_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=f"uturn_{filename}")

@router.get("/event/{track_id}")
async def get_uturn_by_track(track_id: int):
    """
    Get U-turn event details by vehicle track ID.
    """
    records = await db.uturn_records.find({"track_id": track_id}).to_list(length=100)
    if not records:
        raise HTTPException(status_code=404, detail="U-turn not found for track")
    for record in records:
        record["_id"] = str(record["_id"])
    return records
