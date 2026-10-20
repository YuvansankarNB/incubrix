import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.logging_config import logger
from src.api.middleware import ObservabilityMiddleware
from src.api.routes import router as v1_router
from src.pipeline.job_queue import job_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting IncuBrix B-Roll Video Engine...")
    job_manager.start_worker()
    yield
    logger.info("Shutting down IncuBrix B-Roll Video Engine...")

app = FastAPI(
    title="IncuBrix Script-to-B-Roll Video Engine",
    description="High-performance AI engine transforming scripts and voiceovers into license-clean, multi-aspect-ratio B-roll videos.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.yaml",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Idempotency and Correlation Middleware
app.add_middleware(ObservabilityMiddleware)

# Mount API routes under /api/v1
app.include_router(v1_router, prefix="/api/v1")

# Integration Standard #8: Health and Readiness Probes
@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe returning server health status."""
    return {"status": "healthy", "service": "incubrix-broll-engine"}

@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness probe checking internal queues and storage dependencies."""
    storage_ready = Path(settings.LOCAL_STORAGE_DIR).exists()
    return {
        "status": "ready" if storage_ready else "not_ready",
        "dependencies": {
            "storage": "connected" if storage_ready else "error",
            "worker_queue": "operational"
        }
    }

# Global Exception Handler formatting for Integration Standard #5
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during execution.",
                "retryable": True,
                "details": {"exception": str(exc)}
            }
        }
    )

# Static file serving for storage downloads
storage_dir = Path(settings.LOCAL_STORAGE_DIR)
storage_dir.mkdir(parents=True, exist_ok=True)
app.mount("/storage", StaticFiles(directory=str(storage_dir)), name="storage")

# Web Frontend Static Files
web_dir = Path("./src/web")
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

frontend_dir = Path("./frontend")
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

@app.get("/", include_in_schema=False)
async def serve_index():
    dash = frontend_dir / "dashboard.html"
    if dash.exists():
        return FileResponse(str(dash))
    idx = frontend_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    web_idx = web_dir / "index.html"
    if web_idx.exists():
        return FileResponse(str(web_idx))
    return JSONResponse({"message": "IncuBrix Engine running. Open /docs for OpenAPI documentation."})

@app.get("/dashboard", include_in_schema=False)
@app.get("/dashboard.html", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(str(frontend_dir / "dashboard.html"))

@app.get("/create", include_in_schema=False)
@app.get("/create.html", include_in_schema=False)
async def serve_create():
    return FileResponse(str(frontend_dir / "create.html"))

@app.get("/processing", include_in_schema=False)
@app.get("/processing.html", include_in_schema=False)
async def serve_processing():
    return FileResponse(str(frontend_dir / "processing.html"))

@app.get("/results", include_in_schema=False)
@app.get("/results.html", include_in_schema=False)
async def serve_results():
    return FileResponse(str(frontend_dir / "results.html"))

@app.get("/editor", include_in_schema=False)
@app.get("/editor.html", include_in_schema=False)
async def serve_editor():
    return FileResponse(str(frontend_dir / "editor.html"))

@app.get("/settings", include_in_schema=False)
@app.get("/settings.html", include_in_schema=False)
async def serve_settings():
    return FileResponse(str(frontend_dir / "settings.html"))
