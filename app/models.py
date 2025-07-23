from typing import Optional
from pydantic import BaseModel, Field, field_validator
from bson import ObjectId


# ---------------------------------------------------------------------------
# Shared base fields
# ---------------------------------------------------------------------------
class TrafficDataBase(BaseModel):
    vehicle_number: str
    violation_type: str
    timestamp: str  # TODO: upgrade to datetime later if you want


# ---------------------------------------------------------------------------
# Input schema (from client) - no id
# ---------------------------------------------------------------------------
class TrafficDataCreate(TrafficDataBase):
    pass


# ---------------------------------------------------------------------------
# DB/Response schema - exposes Mongo _id as 'id'
# ---------------------------------------------------------------------------
class TrafficDataDB(TrafficDataBase):
    id: Optional[str] = Field(default=None, alias="_id")

    # Convert ObjectId → str before validation
    @field_validator("id", mode="before")
    @classmethod
    def _convert_object_id(cls, v):
        if isinstance(v, ObjectId):
            return str(v)
        if v is None:
            return None
        # accept already-string IDs
        return str(v)

    class Config:
        populate_by_name = True        # allow using field name 'id' OR alias '_id'
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True