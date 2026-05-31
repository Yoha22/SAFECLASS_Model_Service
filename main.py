import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import List

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from config import settings
from model import model
from schemas import (
    Base64PredictRequest,
    CameraStatusResponse,
    FrameQuality,
    HealthResponse,
    PredictResponse,
    StreamActionResponse,
    StreamStartRequest,
    StreamStopRequest,
    ThresholdUpdateRequest,
    ThresholdUpdateResponse,
    VideoAnalyzeResponse,
    VideoStatusResponse,
)
from video_capture import capture_manager
from video_processor import video_job_manager

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_START_TIME = time.monotonic()

# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    logger.info("SAFECLASS model service starting up...")
    model.load_model()
    logger.info("Startup complete. Listening on port %d.", settings.PORT)
    yield
    logger.info("Shutting down model service...")
    model.unload()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SAFECLASS Model Service",
    description=(
        "YOLOv8 inference microservice for SAFECLASS school-safety monitoring. "
        "Detects AGRESION, AISLAMIENTO, CAIDA, and OTRO anomalies in video frames."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Authentication dependency ─────────────────────────────────────────────────


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Dependency: validate X-API-Key header against BACKEND_API_KEY."""
    if not settings.BACKEND_API_KEY:
        # If no key is configured, skip authentication (dev mode)
        return
    if x_api_key != settings.BACKEND_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key.",
        )


# ── Frame helpers ─────────────────────────────────────────────────────────────


def _decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes (JPEG/PNG) into a BGR numpy array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not decode image. Ensure the file is a valid JPEG or PNG.",
        )
    return frame


def _assess_quality(frame: np.ndarray) -> FrameQuality:
    """
    Evaluate frame quality before inference.

    Returns:
        LOW_LIGHT  – mean luminosity < 40
        NOISY      – Laplacian variance < 100 (blurry/noisy)
        OK         – frame passes both checks
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 40.0:
        return FrameQuality.LOW_LIGHT
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if laplacian_var < 100.0:
        return FrameQuality.NOISY
    return FrameQuality.OK


def _resize_if_needed(frame: np.ndarray) -> np.ndarray:
    """Resize frame to 640×640 when dimensions do not already match."""
    h, w = frame.shape[:2]
    if h != 640 or w != 640:
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    return frame


def _run_predict(frame: np.ndarray) -> PredictResponse:
    """Shared prediction logic used by both /predict endpoints."""
    t0 = time.perf_counter()

    quality = _assess_quality(frame)
    if quality != FrameQuality.OK:
        logger.info("Frame quality check failed: %s — skipping inference.", quality)
        return PredictResponse(
            detections=[],
            anomaly_detected=False,
            frame_quality=quality,
            processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    frame = _resize_if_needed(frame)
    detections = model.predict(frame)
    processing_ms = round((time.perf_counter() - t0) * 1000, 2)

    return PredictResponse(
        detections=detections,
        anomaly_detected=len(detections) > 0,
        frame_quality=FrameQuality.OK,
        processing_time_ms=processing_ms,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["monitoring"],
)
async def health() -> HealthResponse:
    """
    Returns service health, model load status, current threshold, and uptime.
    No authentication required.
    """
    return HealthResponse(
        status="ok" if model.is_loaded else "degraded",
        model_loaded=model.is_loaded,
        threshold=model.threshold if model.is_loaded else settings.CONFIDENCE_THRESHOLD,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict anomalies in a single frame (multipart)",
    tags=["inference"],
)
async def predict(
    file: UploadFile = File(..., description="JPEG or PNG image frame"),
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    """
    Accepts an image via multipart/form-data (field name: `file`).

    Steps:
    1. Decode the uploaded image.
    2. Assess frame quality (luminosity + sharpness).
    3. Resize to 640×640 if necessary.
    4. Run YOLOv8 inference.
    5. Return detections with processing time.
    """
    raw = await file.read()
    frame = _decode_image(raw)
    return _run_predict(frame)


@app.post(
    "/predict/base64",
    response_model=PredictResponse,
    summary="Predict anomalies in a single frame (base64 JSON)",
    tags=["inference"],
)
async def predict_base64(
    body: Base64PredictRequest,
    _: None = Depends(verify_api_key),
) -> PredictResponse:
    """
    Accepts a base64-encoded image in a JSON body: `{"image": "<base64>"}`.
    Useful for Node.js callers that prefer JSON over multipart.
    """
    try:
        raw = base64.b64decode(body.image)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid base64 string in 'image' field.",
        )
    frame = _decode_image(raw)
    return _run_predict(frame)


@app.get(
    "/stream/status",
    response_model=List[CameraStatusResponse],
    summary="Get status of all configured camera streams",
    tags=["streaming"],
)
async def stream_status(
    _: None = Depends(verify_api_key),
) -> List[CameraStatusResponse]:
    """Returns the current status, URL, and actual FPS of every registered camera."""
    statuses = capture_manager.get_all_statuses()
    return [CameraStatusResponse(**s) for s in statuses]


@app.post(
    "/stream/start",
    response_model=StreamActionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start RTSP capture for a camera",
    tags=["streaming"],
)
async def stream_start(
    body: StreamStartRequest,
    _: None = Depends(verify_api_key),
) -> StreamActionResponse:
    """
    Starts an asynchronous RTSP capture loop for the given camera.

    Frames are processed at up to MAX_FPS.  Whenever an anomaly is detected,
    the result is forwarded to the backend via POST {BACKEND_URL}/api/internal/alert.
    The endpoint returns immediately (202 Accepted); capture runs in the background.
    """
    capture_manager.start(body.camera_id, body.rtsp_url)
    return StreamActionResponse(
        camera_id=body.camera_id,
        message=f"Capture started for camera '{body.camera_id}'.",
    )


@app.post(
    "/stream/stop",
    response_model=StreamActionResponse,
    summary="Stop RTSP capture for a camera",
    tags=["streaming"],
)
async def stream_stop(
    body: StreamStopRequest,
    _: None = Depends(verify_api_key),
) -> StreamActionResponse:
    """Requests a graceful stop of the capture loop for the specified camera."""
    try:
        capture_manager.stop(body.camera_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{body.camera_id}' is not registered.",
        )
    return StreamActionResponse(
        camera_id=body.camera_id,
        message=f"Stop requested for camera '{body.camera_id}'.",
    )


@app.post(
    "/video/analyze",
    response_model=VideoAnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze a video file for anomalies",
    tags=["video"],
)
async def video_analyze(
    camera_id: str = File(..., description="Camera UUID to associate detections with"),
    video: UploadFile = File(..., description="Video file (mp4, avi, mov, etc.)"),
    _: None = Depends(verify_api_key),
) -> VideoAnalyzeResponse:
    """
    Accepts a video file and a camera_id.
    Processes the video asynchronously frame by frame using the same YOLOv8
    pipeline as live RTSP streams, then forwards each detection to the backend
    via POST /api/internal/alert.
    Returns immediately with a job_id; poll /video/status/{job_id} for progress.
    """
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="El archivo de video está vacío.",
        )

    job_id = video_job_manager.submit(camera_id, video_bytes)
    return VideoAnalyzeResponse(
        job_id=job_id,
        camera_id=camera_id,
        status="pending",
        message=f"Análisis iniciado. Consulta el estado en /video/status/{job_id}",
    )


@app.get(
    "/video/status/{job_id}",
    response_model=VideoStatusResponse,
    summary="Get video analysis job status",
    tags=["video"],
)
async def video_status(
    job_id: str,
    _: None = Depends(verify_api_key),
) -> VideoStatusResponse:
    """Returns the current progress and results of a video analysis job."""
    job_status_data = video_job_manager.get_status(job_id)
    if job_status_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' no encontrado.",
        )
    return VideoStatusResponse(**job_status_data)


@app.put(
    "/config/threshold",
    response_model=ThresholdUpdateResponse,
    summary="Update confidence threshold at runtime",
    tags=["configuration"],
)
async def update_threshold(
    body: ThresholdUpdateRequest,
    _: None = Depends(verify_api_key),
) -> ThresholdUpdateResponse:
    """
    Updates the YOLOv8 confidence threshold without restarting the service (HU-17).
    Accepted range: 0.50 – 0.95.
    Emits a warning in the response if the new value is below 0.60.
    """
    model.threshold = body.threshold
    warning = None
    if body.threshold < 0.60:
        warning = (
            f"Threshold {body.threshold:.2f} is below the recommended minimum of 0.60. "
            "This may generate a high volume of false-positive alerts."
        )
        logger.warning(warning)
    return ThresholdUpdateResponse(threshold=body.threshold, warning=warning)
