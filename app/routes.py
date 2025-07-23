from typing import List
from fastapi import APIRouter, HTTPException, status
from bson import ObjectId

from app.database import db
from app.models import TrafficDataCreate, TrafficDataDB

router = APIRouter()


def _dump(model):
    """Compatibility: Pydantic v1/v2 safe dict export w/ aliases."""
    try:
        return model.model_dump(by_alias=True, exclude_none=True)  # v2
    except AttributeError:
        return model.dict(by_alias=True, exclude_none=True)        # v1 fallback


@router.post(
    "/traffic",
    response_model=TrafficDataDB,
    status_code=status.HTTP_201_CREATED,
)
async def add_traffic_data(data: TrafficDataCreate):
    payload = _dump(data)
    result = await db.traffic_records.insert_one(payload)
    created = await db.traffic_records.find_one({"_id": result.inserted_id})
    if not created:
        raise HTTPException(status_code=500, detail="Insert failed; document not found.")
    return TrafficDataDB(**created)


@router.get(
    "/traffic",
    response_model=List[TrafficDataDB],
    status_code=status.HTTP_200_OK,
)
async def get_traffic_data(limit: int = 100):
    docs = await db.traffic_records.find().limit(limit).to_list(length=limit)
    return [TrafficDataDB(**d) for d in docs]


@router.get(
    "/traffic/{record_id}",
    response_model=TrafficDataDB,
    status_code=status.HTTP_200_OK,
)
async def get_traffic_record(record_id: str):
    if not ObjectId.is_valid(record_id):
        raise HTTPException(status_code=400, detail="Invalid record id.")
    doc = await db.traffic_records.find_one({"_id": ObjectId(record_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Record not found.")
    return TrafficDataDB(**doc)