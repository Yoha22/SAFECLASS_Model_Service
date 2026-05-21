from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ClassName(str, Enum):
    AGRESION    = "AGRESION"
    AISLAMIENTO = "AISLAMIENTO"
    CAIDA       = "CAIDA"
    OTRO        = "OTRO"


class FrameQuality(str, Enum):
    OK         = "OK"
    LOW_LIGHT  = "LOW_LIGHT"
    NOISY      = "NOISY"
    DISCARDED  = "DISCARDED"


class CameraStatus(str, Enum):
    ACTIVE       = "ACTIVE"
    ERROR        = "ERROR"
    RECONNECTING = "RECONNECTING"


# ── Request bodies ────────────────────────────────────────────────────────────

class Base64PredictRequest(BaseModel):
    image: str  # base64-encoded JPEG or PNG


class StreamStartRequest(BaseModel):
    camera_id: str
    rtsp_url: str


class StreamStopRequest(BaseModel):
    camera_id: str


class ThresholdUpdateRequest(BaseModel):
    threshold: float

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.50 <= v <= 0.95):
            raise ValueError("threshold must be between 0.50 and 0.95")
        return v


# ── Response models ───────────────────────────────────────────────────────────

class Detection(BaseModel):
    class_name:  ClassName
    confidence:  float
    bbox:        List[float]   # [x1, y1, x2, y2]
    timestamp:   datetime


class PredictResponse(BaseModel):
    detections:          List[Detection]
    anomaly_detected:    bool
    frame_quality:       FrameQuality
    processing_time_ms:  float
    model_version:       str = "safeclass_best_v1"


class HealthResponse(BaseModel):
    status:          str
    model_loaded:    bool
    threshold:       float
    uptime_seconds:  float


class CameraStatusResponse(BaseModel):
    camera_id:   str
    url:         str
    status:      CameraStatus
    fps_actual:  float


class StreamActionResponse(BaseModel):
    camera_id: str
    message:   str


class ThresholdUpdateResponse(BaseModel):
    threshold: float
    warning:   Optional[str] = None
