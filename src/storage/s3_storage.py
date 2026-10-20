import os
import shutil
import aiofiles
from pathlib import Path
from typing import Optional
from src.config import settings
from src.logging_config import logger

class StorageService:
    def __init__(self):
        self.base_dir = Path(settings.LOCAL_STORAGE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "inputs").mkdir(exist_ok=True)
        (self.base_dir / "outputs").mkdir(exist_ok=True)
        (self.base_dir / "temp").mkdir(exist_ok=True)

    def get_job_dir(self, job_id: str) -> Path:
        job_dir = self.base_dir / "outputs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def save_bytes(self, destination_path: Path, data: bytes):
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with open(destination_path, "wb") as f:
            f.write(data)

    async def save_upload(self, job_id: str, filename: str, content: bytes) -> Path:
        target = self.base_dir / "inputs" / f"{job_id}_{filename}"
        async with aiofiles.open(target, "wb") as f:
            await f.write(content)
        return target

    def get_download_url(self, file_path: Path, base_url: str = "http://localhost:8000") -> str:
        # Generate URL relative to static mounts
        try:
            rel = file_path.relative_to(self.base_dir)
            # Normalize to forward slash for URL
            rel_str = str(rel).replace("\\", "/")
            return f"{base_url}/storage/{rel_str}"
        except ValueError:
            return f"{base_url}/storage/{file_path.name}"

storage = StorageService()
