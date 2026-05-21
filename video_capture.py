import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Awaitable, Dict, Optional

import cv2
import httpx
import numpy as np

from config import settings
from schemas import CameraStatus, PredictResponse

logger = logging.getLogger(__name__)

_MAX_RETRIES    = 3
_RETRY_DELAY_S  = 5.0   # seconds between reconnect attempts
_WEBHOOK_TIMEOUT_S = 5.0


class _CameraState:
    """Runtime state for a single camera capture loop."""

    def __init__(self, camera_id: str, rtsp_url: str) -> None:
        self.camera_id  = camera_id
        self.rtsp_url   = rtsp_url
        self.status     = CameraStatus.RECONNECTING
        self.fps_actual = 0.0
        self.task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()


class CaptureManager:
    """
    Manages asynchronous RTSP capture loops for multiple cameras.

    Each camera runs in its own asyncio Task.  When a frame is processed
    and an anomaly is detected, the result is forwarded to the backend via
    a POST webhook (POST {BACKEND_URL}/api/internal/alert).
    """

    def __init__(self) -> None:
        self._cameras: Dict[str, _CameraState] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self, camera_id: str, rtsp_url: str) -> None:
        """Start asynchronous capture for a camera. Idempotent."""
        if camera_id in self._cameras and not self._cameras[camera_id].should_stop():
            logger.info("Camera %s is already running.", camera_id)
            return

        state = _CameraState(camera_id, rtsp_url)
        self._cameras[camera_id] = state
        state.task = asyncio.create_task(
            self._capture_loop(state),
            name=f"capture-{camera_id}",
        )
        logger.info("Started capture task for camera %s (%s).", camera_id, rtsp_url)

    def stop(self, camera_id: str) -> None:
        """Request graceful stop of a camera's capture loop."""
        state = self._cameras.get(camera_id)
        if state is None:
            raise KeyError(f"Camera '{camera_id}' is not registered.")
        state.request_stop()
        logger.info("Stop requested for camera %s.", camera_id)

    def get_all_statuses(self) -> list[dict]:
        return [
            {
                "camera_id":  s.camera_id,
                "url":        s.rtsp_url,
                "status":     s.status,
                "fps_actual": round(s.fps_actual, 2),
            }
            for s in self._cameras.values()
        ]

    def get_status(self, camera_id: str) -> Optional[dict]:
        state = self._cameras.get(camera_id)
        if state is None:
            return None
        return {
            "camera_id":  state.camera_id,
            "url":        state.rtsp_url,
            "status":     state.status,
            "fps_actual": round(state.fps_actual, 2),
        }

    # ── Capture loop ──────────────────────────────────────────────────────────

    async def _capture_loop(self, state: _CameraState) -> None:
        """Main loop: open stream → read frames → infer → webhook."""
        retries = 0

        while not state.should_stop():
            cap = await asyncio.to_thread(cv2.VideoCapture, state.rtsp_url)

            if not cap.isOpened():
                retries += 1
                logger.warning(
                    "Camera %s: cannot open stream (attempt %d/%d).",
                    state.camera_id, retries, _MAX_RETRIES,
                )
                state.status = CameraStatus.RECONNECTING
                if retries >= _MAX_RETRIES:
                    logger.error(
                        "Camera %s: exceeded max retries, marking as ERROR.",
                        state.camera_id,
                    )
                    state.status = CameraStatus.ERROR
                    return
                await asyncio.sleep(_RETRY_DELAY_S)
                continue

            retries = 0
            state.status = CameraStatus.ACTIVE
            logger.info("Camera %s: stream opened successfully.", state.camera_id)

            frame_interval = 1.0 / settings.MAX_FPS
            last_frame_time = 0.0

            try:
                while not state.should_stop():
                    now = time.monotonic()
                    elapsed = now - last_frame_time

                    # Throttle to MAX_FPS
                    if elapsed < frame_interval:
                        await asyncio.sleep(frame_interval - elapsed)
                        continue

                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret:
                        logger.warning(
                            "Camera %s: failed to read frame, reconnecting...",
                            state.camera_id,
                        )
                        state.status = CameraStatus.RECONNECTING
                        break

                    last_frame_time = time.monotonic()
                    actual_fps = 1.0 / max(elapsed, 1e-6)
                    # Smooth FPS with exponential moving average
                    state.fps_actual = 0.9 * state.fps_actual + 0.1 * actual_fps

                    # Run inference in thread pool (CPU-bound)
                    await self._process_frame(state.camera_id, frame)

            finally:
                cap.release()

        logger.info("Camera %s: capture loop stopped.", state.camera_id)
        state.status = CameraStatus.ERROR  # mark stopped as not active

    # ── Inference + webhook ───────────────────────────────────────────────────

    async def _process_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Run inference and, if anomaly detected, POST to backend webhook."""
        from model import model  # local import to avoid circular dependency
        import time as _time

        t0 = _time.perf_counter()

        try:
            detections = await asyncio.to_thread(model.predict, frame)
        except Exception as exc:
            logger.warning("Camera %s: inference error: %s", camera_id, exc)
            return

        processing_ms = (_time.perf_counter() - t0) * 1000

        if not detections:
            return

        response = PredictResponse(
            detections=detections,
            anomaly_detected=True,
            frame_quality="OK",
            processing_time_ms=round(processing_ms, 2),
        )

        await self._send_webhook(camera_id, response)

    async def _send_webhook(
        self, camera_id: str, response: PredictResponse
    ) -> None:
        """Forward detection result to the backend REST API."""
        url = f"{settings.BACKEND_URL}/api/internal/alert"
        payload = response.model_dump(mode="json")
        payload["camera_id"] = camera_id

        try:
            async with httpx.AsyncClient(timeout=_WEBHOOK_TIMEOUT_S) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "X-API-Key": settings.BACKEND_API_KEY,
                        "Content-Type": "application/json",
                    },
                )
            if resp.status_code not in (200, 201, 204):
                logger.warning(
                    "Webhook to backend returned %d for camera %s.",
                    resp.status_code, camera_id,
                )
            else:
                logger.info(
                    "Alert sent to backend for camera %s (detections=%d).",
                    camera_id, len(response.detections),
                )
        except httpx.RequestError as exc:
            logger.error(
                "Could not reach backend webhook for camera %s: %s",
                camera_id, exc,
            )


# Module-level singleton
capture_manager = CaptureManager()
