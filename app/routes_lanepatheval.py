import os
import shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.lane_change_detection.lane_path_eval import LaneChangeDetector 
from app.utils.video_converter import convert_to_browser_compatible

router = APIRouter(prefix="/lanepatheval", tags=["Lane path Evaluation"])

LANE_MODEL_PATH = "models/lane_model.pt"
VEHICLE_MODEL_PATH = "models/yolov8s.pt"
CONFIDENCE = 0.5

@router.post("/run")
async def run_lane_change_detection(file: UploadFile = File(...)):
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = os.path.join("inputs", file.filename)
    output_path = os.path.join("outputs", f"lanechange_{file.filename}")

    try:
        # Save uploaded video
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Run the NEW detector
        analysis = LaneChangeDetector.run_lane_change_detection(
            video_path=video_path,
            lane_model_path=LANE_MODEL_PATH,
            vehicle_model_path=VEHICLE_MODEL_PATH,
            output_path=output_path,
            confidence=CONFIDENCE,
            save_frequency=1,
            show_display=False
        )

        # Extract event data
        lane_change_events = analysis["lane_change_events"]
        overlay_path = analysis.get("lane_overlay_path", None)

        change_details = []
        for event in lane_change_events:
            # The keys below should match your actual event dictionary
            change_details.append({
                "track_id": event.get("track_id"),
                "timestamp": event.get("timestamp"),
                "frame_idx": event.get("frame_idx"),
                "lane_idx": event.get("lane_idx"),
                "direction": event.get("direction"),
            })

        vehicles_with_lc = list({event["track_id"] for event in change_details})

        vehicles = []
        for v_id in vehicles_with_lc:
            v_changes = [e for e in change_details if e["track_id"] == v_id]
            vehicles.append({
                "vehicle_id": v_id,
                "lane_changes": v_changes,
                "total_lane_changes": len(v_changes),
            })

        summary = {
            "lane_overlay_url": f"http://127.0.0.1:8000/static/{os.path.basename(overlay_path)}" if overlay_path else None,
            "total_detected_lane_changes": len(change_details),
            "vehicles_with_lane_changes": len(vehicles_with_lc),
            "lane_count": analysis.get("lane_count"),
            "processed_at": datetime.utcnow(),
        }

        convert_to_browser_compatible(output_path)
        return {
            "annotated_video_url": f"http://127.0.0.1:8000/static/lanepathval_{file.filename}",
            "summary": summary,
            "vehicles": vehicles,
            "lane_change_events": change_details,
        }
    except Exception as e:
        return {"detail": f"Processing failed: {str(e)}"}

@router.get("/output/{filename}")
async def get_lanechange_output_video(filename: str):
    file_path = os.path.join("outputs", f"lanepathval_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=f"lanepathval_{filename}")

@router.get("/event/{track_id}")
async def get_lane_changes_by_track(track_id: int):
    # (Implement DB query here if needed, or keep as stub)
    raise HTTPException(status_code=501, detail="Not implemented")
