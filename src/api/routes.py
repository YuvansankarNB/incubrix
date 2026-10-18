import uuid
import json
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Header, UploadFile, File, Form, Request, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.models import (
    CreateJobRequest,
    JobCreatedResponse,
    JobStatusResponse,
    SwapClipRequest,
    StandardErrorResponse
)
from src.api.middleware import idempotency_cache
from src.pipeline.job_queue import job_manager
from src.pipeline.audio_engine import AudioEngineException
from src.storage.s3_storage import storage
from src.logging_config import logger

router = APIRouter()

@router.post(
    "/broll/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse,
    responses={
        400: {"model": StandardErrorResponse},
        422: {"model": StandardErrorResponse},
        500: {"model": StandardErrorResponse}
    }
)
async def create_broll_job(
    request: Request,
    body: Optional[CreateJobRequest] = None,
    idempotency_key: str = Header(..., alias="Idempotency-Key")
):
    """
    Submits a new Script / Voiceover to B-Roll rendering job.
    Processes asynchronously and returns HTTP 202 Accepted.
    """
    if not body:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_PAYLOAD", "message": "Request payload is required.", "retryable": False}
        )

    # Validate that at least one narration input exists (check non-empty / non-whitespace)
    has_script = bool(body.script and body.script.strip())
    has_audio_url = bool(body.audio_url and body.audio_url.strip())
    has_scenes = bool(body.scenes and len(body.scenes) > 0)

    if not has_script and not has_audio_url and not has_scenes:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "MISSING_INPUT_CONTENT",
                    "message": "At least one of 'script', 'audio_url', or 'scenes' must be supplied.",
                    "retryable": False,
                    "details": {}
                }
            }
        )

    job_id = f"job_{uuid.uuid4().hex[:16]}"
    job_state = job_manager.create_job(job_id, body.model_dump())

    response_data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": job_state.created_at,
        "message": "Job accepted for background execution"
    }

    # Cache for idempotency
    idempotency_cache[idempotency_key] = {
        "status_code": 202,
        "content": response_data
    }

    return response_data

@router.post(
    "/broll/jobs/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreatedResponse
)
async def create_broll_job_upload(
    file: UploadFile = File(...),
    aspect_ratios: str = Form("16:9"),
    style: str = Form("bright, modern, real-world footage"),
    avoid: str = Form(""),
    burn_in_captions: bool = Form(True),
    idempotency_key: str = Header(..., alias="Idempotency-Key")
):
    """
    Direct audio file upload endpoint supporting .mp3, .wav, .m4a files up to 10 minutes.
    """
    job_id = f"job_{uuid.uuid4().hex[:16]}"
    
    # Save uploaded bytes to local storage inputs
    content = await file.read()
    saved_path = await storage.save_upload(job_id, file.filename, content)

    ratios = [r.strip() for r in aspect_ratios.split(",") if r.strip()]
    avoid_list = [a.strip() for a in avoid.split(",") if a.strip()]

    request_data = {
        "local_audio_path": str(saved_path),
        "aspect_ratios": ratios,
        "style": style,
        "avoid": avoid_list,
        "captions": {"burn_in": burn_in_captions, "srt": True}
    }

    job_state = job_manager.create_job(job_id, request_data)

    response_data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": job_state.created_at,
        "message": "Uploaded audio job accepted for background execution"
    }

    idempotency_cache[idempotency_key] = {
        "status_code": 202,
        "content": response_data
    }

    return response_data

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    responses={
        404: {"model": StandardErrorResponse}
    }
)
async def get_job_status(job_id: str):
    """
    Polls the execution status and output asset links for a given job.
    """
    job_state = job_manager.get_job(job_id)
    if not job_state:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": f"No job exists with ID '{job_id}'.",
                    "retryable": False,
                    "details": {"job_id": job_id}
                }
            }
        )

    return job_state.to_dict()

@router.post(
    "/jobs/{job_id}/swap-clip",
    responses={
        400: {"model": StandardErrorResponse},
        404: {"model": StandardErrorResponse}
    }
)
async def swap_clip_endpoint(job_id: str, body: SwapClipRequest):
    """
    Replaces a single scene's footage and performs a fast partial re-render.
    """
    job_state = job_manager.get_job(job_id)
    if not job_state:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "JOB_NOT_FOUND",
                    "message": f"No job exists with ID '{job_id}'.",
                    "retryable": False,
                    "details": {"job_id": job_id}
                }
            }
        )

    try:
        updated_outputs = await job_manager.swap_clip(
            job_id=job_id,
            scene_index=body.scene_index,
            new_query=body.new_query,
            preferred_source=body.preferred_source
        )
        return {"job_id": job_id, "status": "succeeded", "outputs": updated_outputs}
    except AudioEngineException as ae:
        return JSONResponse(
            status_code=400,
            content={"error": {"code": ae.code, "message": ae.message, "retryable": ae.retryable, "details": ae.details}}
        )
    except Exception as e:
        logger.exception("Error in swap clip")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "SWAP_CLIP_FAILED", "message": str(e), "retryable": True, "details": {}}}
        )
