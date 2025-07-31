from __future__ import annotations
import os
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.services.fraud_detection.fraud_detection import FraudDetectionService, FraudDetectionInput

router = APIRouter(prefix="/fraud", tags=["Fraud Detection"])
_service = FraudDetectionService()

@router.post("/run")
async def run_fraud_detection(input_data: FraudDetectionInput) -> Dict[str, Any]:
    """
    Run fraud detection on vehicle data.
    Expects JSON input with vehicle details, area_type, and speed_limit.
    """
    try:
        # Run fraud detection
        result = await _service.run_fraud_detection(input_data)
        return JSONResponse(content=result)

    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")