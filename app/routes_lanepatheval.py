import os
import shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.lane_change_detection.lane_path_eval import LanePathEval
from app.utils.video_converter import convert_to_browser_compatible

router = APIRouter(prefix="/lanepatheval", tags=["Lane Change Detection"])

# Instantiate LanePathEval with your defaults
_service = LanePathEval(
    lane_model_path="models/lane_model.pt",   # Your best lane segmentation model
    vehicle_model_path="models/yolov8n.pt",   # Your vehicle detection model
    confidence=0.5
)

@router.post("/run")
async def run_lane_change_detection(file: UploadFile = File(...)):
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    video_path = os.path.join("inputs", file.filename)
    output_path = os.path.join("outputs", f"lanepatheval_{file.filename}")

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = _service.run_eval(
            video_path=video_path,
            output_path=output_path,
            save_overlay=True,
            show_display=False,
            save_frequency=1,
        )
        convert_to_browser_compatible(output_path)

        summary = {
            "total_detected_lane_changes": len(result["lane_change_events"]),
            "lane_count": result["lane_count"],
            "processed_at": datetime.utcnow(),
        }
        return {
            "annotated_video_url": f"http://127.0.0.1:8000/static/lanepatheval_{file.filename}",
            "summary": summary,
            "lane_overlay_path": result["overlay_path"],
            "lane_change_events": result["lane_change_events"],
        }
    except Exception as e:
        return {"detail": f"Processing failed: {str(e)}"}

@router.get("/output/{filename}")
async def get_lanepatheval_output_video(filename: str):
    file_path = os.path.join("outputs", f"lanepatheval_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=f"lanepatheval_{filename}")

@router.get("/event/{track_id}")
async def get_lane_changes_by_track(track_id: int):
    raise HTTPException(status_code=501, detail="Not implemented")
