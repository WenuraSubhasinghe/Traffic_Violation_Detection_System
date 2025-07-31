import os
import shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.lane_change_detection.lane_change_detection import LaneChangeDetector
from app.utils.video_converter import convert_to_browser_compatible

router = APIRouter(prefix="/lanechange", tags=["Lane Change Detection"])

# Instantiate detector (adjust path/confidence if needed)
_service = LaneChangeDetector(model_path="yolov8n.pt", confidence=0.5)

@router.post("/run")
async def run_lane_change_detection(file: UploadFile = File(...)):
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = os.path.join("inputs", file.filename)
    output_path = os.path.join("outputs", f"lanechange_{file.filename}")

    try:
        # Save the uploaded video
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run detection and collect events
        lanechange_events = _service.process_video(video_path, output_path, show_display=False)

      

        # Response with event details, including plate_number
        change_details = []
        for event in lanechange_events:
            change_details.append({
                "track_id": event.get("track_id"),
                "timestamp": event.get("timestamp"),
                "frame_idx": event.get("frame_idx"),
                "from_lane": event.get("from_lane"),
                "to_lane": event.get("to_lane"),
                "plate_number": event.get("plate_number", "")
            })

        vehicles_with_lc = list({event["track_id"] for event in change_details})

        vehicles = []
        for v_id in vehicles_with_lc:
            v_changes = [e for e in change_details if e["track_id"] == v_id]
            plates = list({e["plate_number"] for e in v_changes if e.get("plate_number")})
            vehicles.append({
                "vehicle_id": v_id,
                "lane_changes": v_changes,
                "total_lane_changes": len(v_changes),
                "plates": plates,
            })


        summary = {
            "total_detected_lane_changes": len(change_details),
            "total_tracked_vehicles": len(getattr(_service, "vehicle_tracks", {})),
            "vehicles_with_lane_changes": len(vehicles_with_lc),
            "processed_at": datetime.utcnow(),
        }

  # Convert in-place (keep same name)
        convert_to_browser_compatible(output_path)
        return {
            "annotated_video_url": f"http://127.0.0.1:8000/static/lanechange_{file.filename}",
            "summary": summary,
            "vehicles": vehicles,
            "lanechange_events": change_details,
        }
    except Exception as e:
        # Always return a detail key for errors
        return {"detail": f"Processing failed: {str(e)}"}

@router.get("/output/{filename}")
async def get_lanechange_output_video(filename: str):
    file_path = os.path.join("outputs", f"lanechange_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=f"lanechange_{filename}")

@router.get("/event/{track_id}")
async def get_lane_changes_by_track(track_id: int):
    # If you use database storage, implement DB query here
    raise HTTPException(status_code=501, detail="Not implemented")
