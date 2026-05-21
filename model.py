import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import HTTPException

from config import settings
from schemas import ClassName, Detection

logger = logging.getLogger(__name__)

# Maps YOLOv8 class index → ClassName enum
_CLASS_MAP: dict[int, ClassName] = {
    0: ClassName.AGRESION,
    1: ClassName.AISLAMIENTO,
    2: ClassName.CAIDA,
    3: ClassName.OTRO,
}

# Deduplication window: suppress repeated detections of the same class
_DEDUP_WINDOW_SECONDS = 30


class SAFECLASSModel:
    """Singleton wrapper around the YOLOv8 model."""

    _instance: Optional["SAFECLASSModel"] = None

    def __new__(cls) -> "SAFECLASSModel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the YOLOv8 model. Downloads from Google Drive if needed."""
        if self._initialized:
            return

        model_path = Path(settings.MODEL_PATH)

        if not model_path.exists():
            if settings.GDRIVE_FILE_ID:
                logger.info(
                    "Model file not found. Attempting download from Google Drive "
                    "(file_id=%s)...",
                    settings.GDRIVE_FILE_ID,
                )
                self._download_from_gdrive(model_path)
            else:
                raise FileNotFoundError(
                    f"Model file '{model_path}' not found and GDRIVE_FILE_ID is not set. "
                    "Place safeclass_best.pt in the project root or set GDRIVE_FILE_ID."
                )

        try:
            from ultralytics import YOLO

            logger.info("Loading YOLOv8 model from %s...", model_path)
            self._model = YOLO(str(model_path))
            # Warm-up pass to force CUDA/CPU initialization
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)
            logger.info("Model loaded and warmed up successfully.")
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            raise

        # Per-class last-seen timestamps for deduplication
        self._last_seen: dict[ClassName, datetime] = defaultdict(
            lambda: datetime.min
        )
        self._threshold: float = settings.CONFIDENCE_THRESHOLD
        self._initialized = True

    def unload(self) -> None:
        """Release model resources on shutdown."""
        if self._initialized:
            del self._model
            self._initialized = False
            logger.info("Model unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame.

        Applies deduplication: repeated detections of the same class within
        the last 30 seconds are suppressed to avoid alert storms (HU-09).

        Returns a list of Detection objects (may be empty).
        Raises HTTPException(503) if the model is unavailable.
        """
        if not self._initialized:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Service is unavailable.",
            )

        try:
            results = self._model.predict(
                frame,
                conf=self._threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("Inference error: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"Model inference failed: {exc}",
            )

        now = datetime.utcnow()
        dedup_cutoff = now - timedelta(seconds=_DEDUP_WINDOW_SECONDS)
        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = _CLASS_MAP.get(class_id, ClassName.OTRO)

                # Skip if the same class was already reported within the window
                if self._last_seen[class_name] > dedup_cutoff:
                    logger.debug(
                        "Dedup suppressed %s (last seen %s ago)",
                        class_name,
                        now - self._last_seen[class_name],
                    )
                    continue

                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=round(confidence, 4),
                        bbox=[round(v, 2) for v in xyxy],
                        timestamp=now,
                    )
                )
                self._last_seen[class_name] = now

        return detections

    # ── Runtime config ────────────────────────────────────────────────────────

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        logger.info(
            "Confidence threshold updated: %.2f → %.2f", self._threshold, value
        )
        self._threshold = value

    @property
    def is_loaded(self) -> bool:
        return self._initialized

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _download_from_gdrive(destination: Path) -> None:
        try:
            import gdown  # type: ignore

            url = f"https://drive.google.com/uc?id={settings.GDRIVE_FILE_ID}"
            gdown.download(url, str(destination), quiet=False)
            logger.info("Model downloaded to %s", destination)
        except Exception as exc:
            logger.exception("Google Drive download failed: %s", exc)
            raise RuntimeError(
                f"Could not download model from Google Drive: {exc}"
            ) from exc


# Module-level singleton
model = SAFECLASSModel()
