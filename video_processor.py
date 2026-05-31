import asyncio
import logging
import os
import tempfile
import time
import uuid
from enum import Enum
from typing import Dict, Optional

import cv2
import httpx
import numpy as np

from config import settings
from schemas import FrameQuality, PredictResponse

logger = logging.getLogger(__name__)

_WEBHOOK_TIMEOUT_S = 5.0
_DEDUP_WINDOW_S    = 10.0  # suppress same detection class within 10 video-seconds


class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class _VideoJob:
    def __init__(self, job_id: str, camera_id: str) -> None:
        self.job_id           = job_id
        self.camera_id        = camera_id
        self.status           = JobStatus.PENDING
        self.total_frames     = 0
        self.frames_processed = 0
        self.alerts_sent      = 0
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self._last_detection: Dict[str, float] = {}  # class_name -> video timestamp (s)


class VideoJobManager:
    """Manages asynchronous video file analysis jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, _VideoJob] = {}

    def submit(self, camera_id: str, video_bytes: bytes) -> str:
        job_id = str(uuid.uuid4())
        job = _VideoJob(job_id, camera_id)
        self._jobs[job_id] = job
        job.task = asyncio.create_task(
            self._process_video(job, video_bytes),
            name=f"video-job-{job_id[:8]}",
        )
        logger.info("Video job %s submitted for camera %s.", job_id, camera_id)
        return job_id

    def get_status(self, job_id: str) -> Optional[dict]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        progress = (
            round(job.frames_processed / job.total_frames * 100, 1)
            if job.total_frames > 0 else 0.0
        )
        return {
            "job_id":           job.job_id,
            "status":           job.status,
            "progress":         progress,
            "frames_processed": job.frames_processed,
            "total_frames":     job.total_frames,
            "alerts_sent":      job.alerts_sent,
            "error":            job.error,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _process_video(self, job: _VideoJob, video_bytes: bytes) -> None:
        from model import model as yolo_model  # avoid circular import at module load

        job.status = JobStatus.PROCESSING
        tmp_path: Optional[str] = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(video_bytes)
                tmp_path = tmp.name

            cap = await asyncio.to_thread(cv2.VideoCapture, tmp_path)
            if not cap.isOpened():
                job.status = JobStatus.FAILED
                job.error  = "No se pudo abrir el archivo de video."
                return

            video_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            job.total_frames = max(total, 1)

            # Sample at most MAX_FPS frames per second of video
            sample_every = max(1, int(video_fps / settings.MAX_FPS))
            frame_index  = 0

            while True:
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    break

                if frame_index % sample_every == 0:
                    video_time_s = frame_index / video_fps
                    await self._process_frame(job, frame, video_time_s)

                job.frames_processed = min(frame_index + 1, job.total_frames)
                frame_index += 1

            cap.release()
            job.frames_processed = job.total_frames
            job.status = JobStatus.COMPLETED
            logger.info(
                "Video job %s completed. Frames: %d, Alerts sent: %d.",
                job.job_id, frame_index, job.alerts_sent,
            )

        except Exception as exc:
            logger.error("Video job %s failed: %s", job.job_id, exc)
            job.status = JobStatus.FAILED
            job.error  = str(exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _process_frame(
        self, job: _VideoJob, frame: np.ndarray, video_time_s: float
    ) -> None:
        from model import model as yolo_model

        # Quality check (mirrors main.py logic without importing from there)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 40.0:
            return
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 100.0:
            return

        frame_resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        try:
            t0         = time.perf_counter()
            detections = await asyncio.to_thread(yolo_model.predict, frame_resized)
            proc_ms    = round((time.perf_counter() - t0) * 1000, 2)
        except Exception as exc:
            logger.warning("Video job %s: inference error: %s", job.job_id, exc)
            return

        if not detections:
            return

        # Deduplicate: suppress same class within _DEDUP_WINDOW_S of video time
        filtered = []
        for det in detections:
            cls  = det.class_name.value
            last = job._last_detection.get(cls, -_DEDUP_WINDOW_S - 1)
            if video_time_s - last >= _DEDUP_WINDOW_S:
                filtered.append(det)
                job._last_detection[cls] = video_time_s

        if not filtered:
            return

        response = PredictResponse(
            detections=filtered,
            anomaly_detected=True,
            frame_quality=FrameQuality.OK,
            processing_time_ms=proc_ms,
        )
        await self._send_webhook(job, response)

    async def _send_webhook(self, job: _VideoJob, response: PredictResponse) -> None:
        url     = f"{settings.BACKEND_URL}/api/internal/alert"
        payload = response.model_dump(mode="json")
        payload["camera_id"] = job.camera_id

        try:
            async with httpx.AsyncClient(timeout=_WEBHOOK_TIMEOUT_S) as client:
                resp = await client.post(
                    url,
                    json=payload,
                    headers={
                        "X-API-Key":    settings.BACKEND_API_KEY,
                        "Content-Type": "application/json",
                    },
                )
            if resp.status_code in (200, 201, 204):
                job.alerts_sent += 1
            else:
                logger.warning(
                    "Video job %s: webhook returned %d.", job.job_id, resp.status_code
                )
        except httpx.RequestError as exc:
            logger.error("Video job %s: webhook error: %s", job.job_id, exc)


# Module-level singleton
video_job_manager = VideoJobManager()
