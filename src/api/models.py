from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class SceneInput(BaseModel):
    text: str = Field(..., description="Narration line for the scene")
    visual_hint: Optional[str] = Field(default=None, description="Creator hint for B-roll visuals")

class CaptionSettings(BaseModel):
    burn_in: bool = Field(default=True, description="Whether to burn captions directly into the video frames")
    srt: bool = Field(default=True, description="Whether to generate and export an SRT subtitle file")

class CreateJobRequest(BaseModel):
    audio_url: Optional[str] = Field(default=None, description="URL to voiceover audio (mp3, wav, m4a)")
    script: Optional[str] = Field(default=None, description="Raw text script to synthesize voiceover from")
    scenes: Optional[List[SceneInput]] = Field(default=None, description="Optional IncuBrix structured scene list")
    voice: Optional[str] = Field(default="en-US-AriaNeural", description="Voice identifier for TTS")
    aspect_ratios: List[str] = Field(default=["16:9"], description="List of desired aspect ratios: 16:9, 9:16, 1:1")
    style: Optional[str] = Field(default="bright, modern, real-world footage", description="Visual aesthetic guidance")
    avoid: Optional[List[str]] = Field(default_factory=list, description="List of themes or objects to avoid")
    captions: Optional[CaptionSettings] = Field(default_factory=CaptionSettings)
    callback_url: Optional[str] = Field(default=None, description="Webhook URL for async notifications")

class SwapClipRequest(BaseModel):
    scene_index: int = Field(..., description="Zero-based index of the scene to swap")
    new_query: Optional[str] = Field(default=None, description="Custom search query or description for the new clip")
    preferred_source: Optional[str] = Field(default=None, description="Preferred provider: pexels, pixabay, ai_gen, or mock")

class ErrorDetail(BaseModel):
    code: str = Field(..., description="UPPER_SNAKE_CASE error code")
    message: str = Field(..., description="Human-readable description of the error")
    retryable: bool = Field(..., description="Whether a client retry might succeed")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Diagnostic contextual metadata")

class StandardErrorResponse(BaseModel):
    error: ErrorDetail

class JobCreatedResponse(BaseModel):
    job_id: str
    status: str = "queued"
    created_at: str
    message: str = "Job accepted for background execution"

class JobStatusResponse(BaseModel):
    job_id: str
    status: str # queued, running, succeeded, failed
    progress: int # 0 to 100
    current_step: Optional[str] = None
    error: Optional[ErrorDetail] = None
    outputs: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None
