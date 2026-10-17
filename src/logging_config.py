import json
import logging
import sys
from datetime import datetime, timezone
from contextvars import ContextVar
from typing import Optional

job_id_ctx: ContextVar[Optional[str]] = ContextVar("job_id_ctx", default=None)
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id_ctx", default=None)

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "job_id": job_id_ctx.get() or getattr(record, "job_id", None),
            "correlation_id": correlation_id_ctx.get() or getattr(record, "correlation_id", None)
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def setup_logging(log_level: str = "INFO"):
    logger = logging.getLogger("incubrix")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    # Also attach formatter to root logger for consistency
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(JSONFormatter())
        root.addHandler(h)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    return logger

logger = setup_logging()
