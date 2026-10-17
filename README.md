# IncuBrix Script & Voiceover to B-Roll Video Engine

An enterprise-ready, containerized microservice that turns voiceover audio or text scripts into broadcast-quality, license-clean, multi-aspect-ratio B-roll videos, timed scene-by-scene to the narration.

---

## Architecture Overview

```
 [Client / HTML5 UI / CLI Batch Runner]
                   │
                   ▼ (HTTP API / OpenAPI 3.1)
 [FastAPI Core Application & Idempotency Layer]
         │                          │
         ▼                          ▼
 [Async Worker Pipeline]   [MinIO / S3 Storage Engine]
   1. Multi-lingual Speech & Word-Level STT (Edge-TTS + Faster-Whisper: EN, HI, TA)
   2. Contextual Scene Segmentation & Polysemy Disambiguation
   3. Dual-Footage Arbitration (Pexels / Pixabay / Kinetic Procedural Motion Engine)
   4. Animated AI Character Host (Lip-sync phonemes, blinking eyes, chart gestures)
   5. VLM Semantic Relevance Scoring (Visual feature cosine calibration)
   6. FFmpeg Video Compositor (16:9, 9:16 Smart Crop, 1:1, Color Norm, Burn-in SRT)
   7. Fast Single-Scene Clip Swapper & Audit-Ready Manifest Builder
```

---

## Setup & Execution (Under 10 Commands)

### 1. Local Quickstart (Zero API Keys Required)
```bash
# 1. Clone repository
git clone https://github.com/IncuBrix/broll-engine.git && cd broll-engine

# 2. Copy environment configuration
cp .env.example .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated tests (Unit, Integration, and E2E)
pytest

# 5. Launch API and Web Dashboard
python run_server.py
```
Open your browser at **`http://localhost:8000`** to interact with the HTML/CSS web dashboard or **`http://localhost:8000/docs`** for interactive OpenAPI documentation.

### 2. Docker Compose
```bash
docker compose up --build
```

---

## CLI Batch Runner

Integration Standard #10 verification command:
```bash
python batch_runner.py --input ./test_pack/inputs --output ./test_pack/outputs --api http://localhost:8000
```
Processes every input file in batch mode, polls completion, and downloads all videos, `timeline.json`, `captions.srt`, and `manifest.json` into organized subdirectories.

---

## API Specification

All business endpoints are strictly under `/api/v1/`:

*   `POST /api/v1/broll/jobs` — Submit text script, audio URL, or IncuBrix scene list (HTTP 202 Accepted). Requires `Idempotency-Key` header.
*   `POST /api/v1/broll/jobs/upload` — Direct audio file upload (.mp3, .wav, .m4a up to 10 min).
*   `GET /api/v1/jobs/{job_id}` — Query job progress (0–100%) and retrieve download links.
*   `POST /api/v1/jobs/{job_id}/swap-clip` — Swap an individual scene's footage with fast partial re-rendering.
*   `GET /health` — Liveness check.
*   `GET /ready` — Readiness check.

---

## Error Codes Catalog (Integration Standard #5)

All error responses strictly follow `{"error": {"code", "message", "retryable", "details"}}`:

| Error Code | HTTP Status | Retryable | Description |
|---|---|---|---|
| `MISSING_IDEMPOTENCY_KEY` | 400 | No | The mandatory `Idempotency-Key` header was omitted from a state-changing POST. |
| `MISSING_INPUT_CONTENT` | 400 | No | None of `script`, `audio_url`, or `scenes` were provided in the payload. |
| `EMPTY_SCRIPT_ERROR` | 400 | No | Text script provided was empty or whitespace only. |
| `INVALID_AUDIO_FORMAT` | 400 | No | Audio file could not be parsed or has 0 bytes. |
| `AUDIO_EXCEEDS_MAX_DURATION`| 400 | No | Audio length exceeds the 10-minute (600s) maximum ceiling. |
| `SILENT_AUDIO_DETECTED` | 400 | No | Input audio mean volume is below -55dB (silent or inaudible). |
| `NO_SPEECH_DETECTED` | 400 | No | Whisper detected no spoken speech (music-only or instrumental track). |
| `JOB_NOT_FOUND` | 404 | No | Requested `job_id` does not exist. |
| `JOB_NOT_READY_FOR_SWAP` | 400 | No | Cannot swap clip until the job reaches `succeeded` status. |
| `INVALID_SCENE_INDEX` | 400 | No | Requested scene index is outside the timeline scene count. |
| `SWAP_CLIP_FAILED` | 500 | Yes | Internal error while re-rendering the swapped scene segment. |
| `INTERNAL_SERVER_ERROR` | 500 | Yes | Unhandled exception occurred during background pipeline execution. |

---

## Environment Settings (.env)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port for the HTTP server |
| `HOST` | `0.0.0.0` | Host binding interface |
| `ENVIRONMENT` | `development` | Runtime environment mode |
| `LOG_LEVEL` | `INFO` | Logging granularity (DEBUG, INFO, WARN, ERROR) |
| `SECRET_KEY` | - | HMAC-SHA256 signing secret for webhooks |
| `PEXELS_API_KEY` | `None` | Pexels Video API commercial access key |
| `PIXABAY_API_KEY` | `None` | Pixabay Video API commercial access key |
| `WHISPER_MODEL_SIZE` | `base` | Model size for Faster-Whisper alignment |
| `LOCAL_STORAGE_DIR` | `./data/storage` | Filesystem path for outputs and temporary assets |
| `MIN_SCENE_DURATION_SECONDS`| `1.8` | Pacing threshold to avoid flickering 0.5s cuts |
| `MAX_SCENE_DURATION_SECONDS`| `7.0` | Ceiling threshold before forcing a semantic cut |
