import uuid
import json
from typing import Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from src.logging_config import correlation_id_ctx, job_id_ctx, logger

# In-memory idempotency store (Key -> Cached response dict)
idempotency_cache: Dict[str, Dict[str, Any]] = {}

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract or generate Correlation ID
        corr_id = request.headers.get("X-Correlation-Id") or str(uuid.uuid4())
        correlation_id_ctx.set(corr_id)
        
        # Check Idempotency for POST requests
        if request.method == "POST" and request.url.path.startswith("/api/v1/broll"):
            idempotency_key = request.headers.get("Idempotency-Key")
            if not idempotency_key:
                # Return standard error if Idempotency-Key is missing
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "code": "MISSING_IDEMPOTENCY_KEY",
                            "message": "Header 'Idempotency-Key' is required for all state-changing POST requests.",
                            "retryable": False,
                            "details": {"required_header": "Idempotency-Key"}
                        }
                    }
                )
            
            # Check if request has already been cached
            if idempotency_key in idempotency_cache:
                logger.info(f"Idempotent hit for key: {idempotency_key}")
                cached = idempotency_cache[idempotency_key]
                return JSONResponse(
                    status_code=cached["status_code"],
                    content=cached["content"],
                    headers={"X-Correlation-Id": corr_id, "X-Cache": "HIT"}
                )

        response = await call_next(request)
        response.headers["X-Correlation-Id"] = corr_id
        return response
