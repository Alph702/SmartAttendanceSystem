from typing import List, Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    name: str
    image: str  # Base64 string


class RecognizeRequest(BaseModel):
    image: str  # Base64 string


class AttendanceMatch(BaseModel):
    box: List[int]
    name: str
    similarity: float
    newly_marked: bool


class RecognizeResponse(BaseModel):
    success: bool
    matches: List[AttendanceMatch]
    attendance_error: Optional[str] = None
