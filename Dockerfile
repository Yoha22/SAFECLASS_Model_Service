FROM python:3.11-slim

# ── System dependencies required by OpenCV ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (separate layer for cache efficiency) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Model weights ──────────────────────────────────────────────────────────────
# safeclass_best.pt must exist in the build context (project root) before
# running "docker build".  The file is intentionally excluded from git
# (.gitignore lists *.pt).  Download it from the shared Drive link and place
# it in the safeclass-model-service/ directory, then build.
COPY safeclass_best.pt .

# ── Application source ────────────────────────────────────────────────────────
COPY config.py        .
COPY schemas.py       .
COPY model.py         .
COPY video_capture.py .
COPY main.py          .

EXPOSE 8001

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will mark the container unhealthy if /health stops responding.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
