from typing import List, Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    name: str
    image: str  # Base64 string


class RecognizeRequest(BaseModel):
    image: str  # Base64 string
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    accuracy: Optional[float] = None


class AttendanceMatch(BaseModel):
    box: List[int]
    name: str
    similarity: float
    newly_marked: bool


class RecognizeResponse(BaseModel):
    success: bool
    matches: List[AttendanceMatch]
    attendance_error: Optional[str] = None
    school_lat: Optional[float] = None
    school_lon: Optional[float] = None
    school_radius: Optional[float] = None
    current_accuracy: Optional[float] = None
    max_accuracy: Optional[float] = None
