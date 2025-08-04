import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.services.speed_detection.speed_detection import SpeedViolationDetector
from app.database import db

router = APIRouter(prefix="/speed", tags=["Speed Detection"])
_service = SpeedViolationDetector()

@router.post("/run")
async def run_speed_detection(file: UploadFile = File(...)):
    """
    Upload a video and run speed violation detection.
    """
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_path = os.path.join("inputs", file.filename)
    output_path = os.path.join("outputs", f"annotated_{file.filename}")

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = await _service.process_video(video_path, output_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Video processing failed result is none")

        # Extract violation details from result
        violation_details = []
        if "violations" in result:
            for violation in result["violations"]:
                vehicle_key, speed, speed_limit, is_violation, timestamp = violation
                if is_violation:  # Only include actual violations
                    violation_details.append({
                        "vehicle_key": vehicle_key,
                        "vehicle_id": vehicle_key.split('_')[1],
                        "vehicle_type": vehicle_key.split('_')[0],
                        "speed": round(speed, 1),
                        "speed_limit": speed_limit,
                        "timestamp": timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                        "excess_speed": round(speed - speed_limit, 1)
                    })

        # Extract vehicle details with speed information
        vehicle_details = []
        for vehicle in result["vehicle_data"]:
            vehicle_violations = [v for v in violation_details if v["vehicle_id"] == str(vehicle["vehicle_id"])]
            max_speed = max([s["speed"] for s in vehicle["speed_series"]], default=0)
            avg_speed = sum([s["speed"] for s in vehicle["speed_series"]]) / len(vehicle["speed_series"]) if vehicle["speed_series"] else 0
            
            vehicle_details.append({
                "vehicle_id": vehicle["vehicle_id"],
                "vehicle_type": vehicle["speed_series"][0]["vehicle_type"] if vehicle["speed_series"] else "unknown",
                "max_speed": round(max_speed, 1),
                "avg_speed": round(avg_speed, 1),
                "total_violations": len(vehicle_violations),
                "violations": vehicle_violations
            })

        # Construct URLs for output files
        video_filename = os.path.basename(result["summary"]["video_path"])
        ref_points_filename = os.path.basename(result["reference_points_image"]) if result["reference_points_image"] else None
        transformed_filename = os.path.basename(result["transformed_image"]) if result["transformed_image"] else None

        return {
            "annotated_video_url": f"http://127.0.0.1:8000/static/{video_filename}",
            "reference_points_image_url": f"http://127.0.0.1:8000/static/{ref_points_filename}" if ref_points_filename else None,
            "transformed_image_url": f"http://127.0.0.1:8000/static/{transformed_filename}" if transformed_filename else None,
            "summary": {
                **result["summary"],
                "violation_summary": {
                    "total_violations": len(violation_details),
                    "vehicles_with_violations": len([v for v in vehicle_details if v["total_violations"] > 0]),
                    "average_excess_speed": round(
                        sum([v["excess_speed"] for v in violation_details]) / len(violation_details), 1
                    ) if violation_details else 0
                }
            },
            "vehicles": vehicle_details,
            "violations": violation_details
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/output/{filename}")
async def get_output_video(filename: str):
    """
    Download the annotated video.
    """
    file_path = os.path.join("outputs", f"annotated_{filename}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=f"annotated_{filename}")


@router.get("/vehicle/{vehicle_id}")
async def get_vehicle_speed(vehicle_id: int):
    """
    Fetch speed time-series and violations for a given vehicle ID.
    """
    record = await db.speed_records.find_one({"vehicle_id": vehicle_id})
    if not record:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    record["_id"] = str(record["_id"])  # Convert ObjectId for JSON
    return record