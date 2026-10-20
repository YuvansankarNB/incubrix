import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "incubrix-super-secret-key-production-ready"

    # Footage Provider Keys
    PEXELS_API_KEY: Optional[str] = Field(default=None)
    PIXABAY_API_KEY: Optional[str] = Field(default=None)

    # Background Image APIs (for context-relevant HD photographic backgrounds)
    UNSPLASH_ACCESS_KEY: Optional[str] = Field(default=None)  # https://unsplash.com/developers

    # Talking Avatar / Lip-Sync APIs (for high-quality animated character generation)
    DID_API_KEY: Optional[str] = Field(default=None)          # https://www.d-id.com
    SIMLI_API_KEY: Optional[str] = Field(default=None)        # https://www.simli.com
    HEYGEN_API_KEY: Optional[str] = Field(default=None)       # https://www.heygen.com

    # LLM Settings
    LLM_PROVIDER: str = "rule_based" # 'rule_based', 'openai', 'gemini', 'mock'
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # Audio Engine (Male & Female Character Voice Defaults)
    TTS_DEFAULT_VOICE_EN_MALE: str = "en-US-ChristopherNeural"
    TTS_DEFAULT_VOICE_EN_FEMALE: str = "en-US-JennyNeural"
    TTS_DEFAULT_VOICE_EN: str = "en-US-ChristopherNeural"
    TTS_DEFAULT_VOICE_HI_MALE: str = "hi-IN-MadhurNeural"
    TTS_DEFAULT_VOICE_HI_FEMALE: str = "hi-IN-SwaraNeural"
    TTS_DEFAULT_VOICE_HI: str = "hi-IN-SwaraNeural"
    TTS_DEFAULT_VOICE_TA_MALE: str = "ta-IN-ValluvarNeural"
    TTS_DEFAULT_VOICE_TA_FEMALE: str = "ta-IN-PallaviNeural"
    TTS_DEFAULT_VOICE_TA: str = "ta-IN-PallaviNeural"
    WHISPER_MODEL_SIZE: str = "base"

    # Storage Settings
    STORAGE_MODE: str = "local" # 'local' or 's3'
    LOCAL_STORAGE_DIR: str = os.path.abspath("./data/storage")
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "incubrix-broll"

    # Concurrency & Safety
    MAX_JOB_CONCURRENCY: int = 4
    MAX_AUDIO_DURATION_SECONDS: int = 600
    MIN_SCENE_DURATION_SECONDS: float = 1.8
    MAX_SCENE_DURATION_SECONDS: float = 7.0
    VLM_EVALUATION_ENABLED: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
