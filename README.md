# safeclass-model-service

Microservicio de inferencia YOLOv8 para el sistema SAFECLASS de monitoreo de seguridad escolar.

---

## 1. Descripción y rol en SAFECLASS

Este servicio es el núcleo de detección de anomalías del sistema. Recibe frames de video desde el backend Node.js, ejecuta el modelo YOLOv8 entrenado (`safeclass_best.pt`) y devuelve un JSON con las detecciones.

```
[Cámaras IP RTSP]
      ↓ frames
[safeclass-model-service]  ← ESTE SERVICIO (Python · FastAPI · :8001)
      ↓ JSON detecciones / webhook
[safeclass-backend]        (Node.js · Express · PostgreSQL · :3000)
      ↓ SSE / WebSocket
[safeclass-frontend]       (React · Vite · :5173)
```

### Clases detectadas

| class_id | Nombre | Descripción |
|---|---|---|
| 0 | `AGRESION` | Confrontación física entre estudiantes |
| 1 | `AISLAMIENTO` | Estudiante separado del grupo de forma prolongada |
| 2 | `CAIDA` | Caída brusca al suelo |
| 3 | `OTRO` | Comportamiento inusual no clasificado |

### Métricas del modelo entrenado

| Métrica | Valor |
|---|---|
| mAP@0.5 | 90.4% |
| Precisión | 98.7% |
| Recall | 81.4% |
| F1-score | 0.88 |
| Umbral por defecto | 0.75 |
| Velocidad objetivo | ≥ 15 FPS |
| Tiempo máx. detección → alerta | 2 segundos |

---

## 2. Prerrequisitos

| Herramienta | Versión mínima |
|---|---|
| Python | 3.11 |
| pip | 23.x |
| Docker | 24.x (solo si usas contenedor) |
| Docker Compose | 2.x (solo si usas contenedor) |

---

## 3. Setup local (sin Docker)

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd safeclass-model-service

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Colocar el modelo en la raíz del proyecto
#    Descárgalo del Drive compartido y nómbralo:
cp /ruta/a/tu/modelo/safeclass_best.pt .

# 5. Configurar variables de entorno
cp .env.example .env
# Edita .env con tus valores (BACKEND_API_KEY, BACKEND_URL, etc.)

# 6. Levantar el servicio
uvicorn main:app --reload --port 8001
```

El servicio queda disponible en `http://localhost:8001`.  
Documentación interactiva (Swagger): `http://localhost:8001/docs`.

---

## 4. Setup con Docker

> **Importante:** El archivo `safeclass_best.pt` debe existir en el directorio `safeclass-model-service/` **antes** de ejecutar `docker build`. El modelo no está en git y se copia dentro de la imagen durante el build.

```bash
# 1. Colocar el modelo en la raíz del proyecto
cp /ruta/a/tu/modelo/safeclass_best.pt .

# 2. Copiar y editar variables de entorno
cp .env.example .env

# 3a. Build + run con docker compose (recomendado)
docker compose up --build -d

# 3b. O manualmente:
docker build -t safeclass-model-service .
docker run -p 8001:8001 --env-file .env safeclass-model-service
```

### Verificar que el contenedor está activo

```bash
curl http://localhost:8001/health
# → {"status":"ok","model_loaded":true,"threshold":0.75,"uptime_seconds":12.3}
```

---

## 5. Endpoints — Referencia completa

### `GET /health`

Sin autenticación. Retorna estado del servicio.

```bash
curl http://localhost:8001/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "threshold": 0.75,
  "uptime_seconds": 143.8
}
```

---

### `POST /predict`

Inferencia sobre un frame como `multipart/form-data`.

```bash
curl -X POST http://localhost:8001/predict \
  -H "X-API-Key: tu_api_key" \
  -F "file=@/ruta/al/frame.jpg"
```

**Respuesta exitosa (anomalía detectada):**

```json
{
  "detections": [
    {
      "class_name": "AGRESION",
      "confidence": 0.9231,
      "bbox": [120.5, 80.2, 450.3, 380.1],
      "timestamp": "2026-05-21T14:35:22.104Z"
    }
  ],
  "anomaly_detected": true,
  "frame_quality": "OK",
  "processing_time_ms": 48.7,
  "model_version": "safeclass_best_v1"
}
```

**Respuesta con frame de baja calidad (sin inferencia, no es error):**

```json
{
  "detections": [],
  "anomaly_detected": false,
  "frame_quality": "LOW_LIGHT",
  "processing_time_ms": 3.2,
  "model_version": "safeclass_best_v1"
}
```

---

### `POST /predict/base64`

Inferencia con imagen codificada en base64 (JSON).

```bash
curl -X POST http://localhost:8001/predict/base64 \
  -H "X-API-Key: tu_api_key" \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_string>"}'
```

---

### `GET /stream/status`

Estado de todas las cámaras registradas.

```bash
curl -H "X-API-Key: tu_api_key" http://localhost:8001/stream/status
```

```json
[
  {
    "camera_id": "cam_01",
    "url": "rtsp://192.168.1.100:554/stream",
    "status": "ACTIVE",
    "fps_actual": 14.8
  }
]
```

---

### `POST /stream/start`

Inicia captura asíncrona de una cámara RTSP.

```bash
curl -X POST http://localhost:8001/stream/start \
  -H "X-API-Key: tu_api_key" \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam_01", "rtsp_url": "rtsp://192.168.1.100:554/stream"}'
```

```json
{
  "camera_id": "cam_01",
  "message": "Capture started for camera 'cam_01'."
}
```

---

### `POST /stream/stop`

Detiene la captura de una cámara.

```bash
curl -X POST http://localhost:8001/stream/stop \
  -H "X-API-Key: tu_api_key" \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "cam_01"}'
```

---

### `PUT /config/threshold`

Actualiza el umbral de confianza en caliente (sin reiniciar).

```bash
curl -X PUT http://localhost:8001/config/threshold \
  -H "X-API-Key: tu_api_key" \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.80}'
```

```json
{
  "threshold": 0.80,
  "warning": null
}
```

---

## 6. Cómo el backend Node.js debe llamar a `/predict`

### Con multipart/form-data (axios + form-data)

```js
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function detectAnomalies(frameBuffer) {
  const form = new FormData();
  form.append('file', frameBuffer, {
    filename: 'frame.jpg',
    contentType: 'image/jpeg',
  });

  const response = await axios.post(
    'http://localhost:8001/predict',
    form,
    {
      headers: {
        ...form.getHeaders(),
        'X-API-Key': process.env.MODEL_SERVICE_API_KEY,
      },
      timeout: 5000,
    }
  );

  return response.data; // PredictResponse
}
```

### Con base64 (fetch nativo, Node 18+)

```js
async function detectAnomaliesBase64(frameBuffer) {
  const base64Image = frameBuffer.toString('base64');

  const response = await fetch('http://localhost:8001/predict/base64', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': process.env.MODEL_SERVICE_API_KEY,
    },
    body: JSON.stringify({ image: base64Image }),
  });

  if (!response.ok) {
    throw new Error(`Model service error: ${response.status}`);
  }

  return await response.json(); // PredictResponse
}
```

### Estructura del PredictResponse recibido

```js
{
  detections: [
    {
      class_name: 'AGRESION',   // "AGRESION" | "AISLAMIENTO" | "CAIDA" | "OTRO"
      confidence: 0.9231,        // float 0.0–1.0
      bbox: [120.5, 80.2, 450.3, 380.1], // [x1, y1, x2, y2]
      timestamp: '2026-05-21T14:35:22.104Z'
    }
  ],
  anomaly_detected: true,        // boolean
  frame_quality: 'OK',           // "OK" | "LOW_LIGHT" | "NOISY" | "DISCARDED"
  processing_time_ms: 48.7,
  model_version: 'safeclass_best_v1'
}
```

---

## 7. Variables de entorno

| Variable | Por defecto | Requerida | Descripción |
|---|---|---|---|
| `MODEL_PATH` | `safeclass_best.pt` | Sí | Ruta al archivo de pesos YOLOv8 |
| `CONFIDENCE_THRESHOLD` | `0.75` | No | Umbral de confianza (0.50 – 0.95) |
| `MAX_FPS` | `15` | No | FPS máximos por cámara RTSP |
| `GDRIVE_FILE_ID` | `""` | No | ID del archivo en Google Drive para descarga automática |
| `BACKEND_API_KEY` | `""` | Recomendada | Clave de autenticación X-API-Key |
| `BACKEND_URL` | `http://localhost:3000` | No | URL del backend Node.js (para webhooks) |
| `PORT` | `8001` | No | Puerto del servidor FastAPI |

---

## 8. Cómo actualizar el modelo

Cuando se disponga de un nuevo archivo de pesos entrenado:

```bash
# 1. Reemplazar el archivo de pesos con el nuevo modelo
cp /ruta/al/nuevo/modelo/safeclass_best_v2.pt safeclass_best.pt

# 2. Si cambia el nombre del archivo, actualizar MODEL_PATH en .env
#    MODEL_PATH=safeclass_best_v2.pt

# 3. Reconstruir la imagen Docker (los pesos se copian durante el build)
docker compose build --no-cache

# 4. Reiniciar el contenedor
docker compose up -d

# 5. Verificar que el nuevo modelo cargó correctamente
curl http://localhost:8001/health
```

> El campo `model_version` en `PredictResponse` actualmente devuelve `"safeclass_best_v1"`.
> Actualiza la constante en `schemas.py` → `PredictResponse.model_version` al desplegar un nuevo modelo.
