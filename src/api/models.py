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
    topic: Optional[str] = Field(default=None, description="Topic to research and create a script from")
    scenes: Optional[List[Union[SceneInput, Dict[str, Any]]]] = Field(default=None, description="Optional structured scene list")
    voice: Optional[str] = Field(default=None, description="Voice identifier for TTS, or None for auto character-matched voice")
    voice_gender: Optional[str] = Field(default="auto", description="Voice gender preference: auto, male, female")
    aspect_ratios: List[str] = Field(default=["16:9"], description="List of desired aspect ratios: 16:9, 9:16, 1:1")
    style: Optional[str] = Field(default="bright, modern, real-world footage", description="Visual aesthetic guidance")
    visual_source_preference: Optional[str] = Field(default="stock_first", description="Visual source selection policy: stock_first (default Pixabay), hybrid, ai_first")
    target_duration: Optional[float] = Field(default=None, description="Target video duration in seconds")
    avoid: Optional[List[str]] = Field(default_factory=list, description="List of themes or objects to avoid")
    captions: Optional[CaptionSettings] = Field(default_factory=CaptionSettings)
    character_type: Optional[str] = Field(default="human", description="Character type to render in videos: human, doctor, patient, teacher, student, executive, narrator, farmer, etc.")
    research_mode: Optional[str] = Field(default="script_plus_research", description="Research mode: 'script_only', 'script_plus_research', or 'research_first'")
    is_voiceover: Optional[bool] = Field(default=False, description="When True, character mouth remains closed during voiceover narration; when False, mouth lip-syncs to dialogue")
    animation_style: Optional[str] = Field(default="3d_realistic", description="Character visual style: 3d_realistic, stylized, cinematic")
    text_style: Optional[str] = Field(default="context_aware", description="Typography style: context_aware, minimal_glass, bold_broadcast")
    show_character_labels: Optional[bool] = Field(default=False, description="Whether to display character nameplates and role badges below characters (defaults to False)")
    callback_url: Optional[str] = Field(default=None, description="Webhook URL for async notifications")

class GenerateScenePlanRequest(BaseModel):
    script: Optional[str] = Field(default=None, description="Raw script text")
    topic: Optional[str] = Field(default=None, description="Topic for researched script generation")
    scenes: Optional[List[Union[SceneInput, Dict[str, Any]]]] = Field(default=None, description="Draft scene list")
    character_type: Optional[str] = Field(default="human", description="Character archetype")
    style: Optional[str] = Field(default="bright, modern, real-world footage", description="Visual style")
    target_duration: Optional[float] = Field(default=None, description="Desired total video duration")
    research_mode: Optional[str] = Field(default="script_plus_research", description="Research mode")

class ScenePlanResponse(BaseModel):
    scenes: List[Dict[str, Any]]
    total_duration: float
    research_bundle: Optional[Dict[str, Any]] = None
    character_type: str

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

# ──────────────────────────────────────────────
# Pixabay Integration Models
# ──────────────────────────────────────────────

class PixabayVideoResultModel(BaseModel):
    """Single Pixabay video search hit (API response item)."""
    pixabay_id: int
    page_url: str
    user: str
    user_id: int
    user_profile_url: str
    tags: str
    duration: int
    thumbnail_url: str
    download_url: str
    width: int
    height: int
    size: int
    quality_tier: str
    editors_choice: bool
    views: int
    downloads: int
    relevance_score: float = 0.0
    is_mock: bool = False

class PixabayVideoSearchResponse(BaseModel):
    """Response from GET /api/v1/stock/pixabay/videos"""
    query: str
    page: int
    per_page: int
    aspect_ratio: str
    total_returned: int
    results: List[PixabayVideoResultModel]

class SceneClipSearchRequest(BaseModel):
    """Request body for POST /api/v1/jobs/{job_id}/scenes/{scene_id}/search-clips"""
    query: Optional[str] = Field(default=None, description="Search query; if omitted, derived from scene narration")
    per_page: int = Field(default=12, ge=3, le=50, description="Number of alternatives to return")
    aspect_ratio: Optional[str] = Field(default=None, description="Override aspect ratio: 16:9, 9:16, 1:1")
    target_duration: Optional[float] = Field(default=None, description="Target clip duration in seconds")
    editors_choice: bool = Field(default=False, description="Prefer Pixabay editors-choice clips")

class SceneClipReplaceRequest(BaseModel):
    """Request body for scene clip replacement (alias for SwapClipRequest, scene_id in path)."""
    new_query: Optional[str] = Field(default=None, description="New search query for replacement clip")
    preferred_source: Optional[str] = Field(default="pixabay", description="Preferred provider: pexels, pixabay, ai_gen, mock")
    pixabay_video_id: Optional[int] = Field(default=None, description="If set, use this specific Pixabay video ID")

class JobSourcesEntry(BaseModel):
    scene_index: int
    scene_narration: Optional[str] = None
    asset_id: Optional[str] = None
    provider: Optional[str] = None
    page_url: Optional[str] = None
    creator: Optional[str] = None
    license_name: Optional[str] = None
    license_url: Optional[str] = None
    commercial_use_eligible: Optional[bool] = None
    is_mock: Optional[bool] = None
    download_date: Optional[str] = None
    pixabay_video_id: Optional[int] = None
    thumbnail_url: Optional[str] = None

class JobSourcesResponse(BaseModel):
    """Response from GET /api/v1/jobs/{job_id}/sources"""
    job_id: str
    total_scenes: int
    sources: List[JobSourcesEntry]

