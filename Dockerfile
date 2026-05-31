# Single-stage: python:3.11-slim
# PyTorch / ultralytics ya distribuyen wheels pre-compilados en PyPI;
# compilar wheels localmente es innecesario y agota recursos en Docker Desktop.
FROM python:3.11-slim

LABEL maintainer="safeclass-team" \
      service="safeclass-model" \
      version="1.0.0"

# Runtime libraries requeridas por OpenCV y ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Usuario sin privilegios
RUN useradd -r -s /sbin/nologin -d /app safeclass

WORKDIR /app

# Dependencias Python — torch CPU-only primero para evitar descargar ~2 GB de CUDA
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall \
        torch==2.5.1+cpu \
        torchvision==0.20.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu \
    && rm requirements.txt

# Pesos del modelo (baked en la imagen — reconstruir para actualizar)
COPY safeclass_best.pt .

# Código fuente
COPY config.py schemas.py model.py video_capture.py video_processor.py main.py ./

RUN chown -R safeclass:safeclass /app

USER safeclass

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
