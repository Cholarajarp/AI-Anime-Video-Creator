"""
Pydantic Models for Jobs and API
================================
Type-safe data models for the application.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Possible states for a generation job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority levels."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class GenerationSettings(BaseModel):
    """Video generation settings."""
    width: int = Field(default=512, ge=256, le=1024, description="Video width")
    height: int = Field(default=512, ge=256, le=1024, description="Video height")
    fps: int = Field(default=15, ge=8, le=30, description="Frames per second")
    steps: int = Field(default=20, ge=10, le=50, description="Diffusion steps")
    cfg_scale: float = Field(default=7.0, ge=1.0, le=20.0, description="CFG scale")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    checkpoint: str = Field(default="dreamshaper_8.safetensors", description="Model checkpoint")
    motion_module: str = Field(default="mm_sd_v15_v2.ckpt", description="AnimateDiff motion module")


class CreateJobRequest(BaseModel):
    """Request to create a new generation job."""
    prompt: str = Field(..., min_length=1, max_length=2000, description="Visual prompt")
    negative_prompt: str = Field(
        default="lowres, bad anatomy, bad hands, text, error",
        max_length=1000,
        description="Negative prompt"
    )
    script: str = Field(..., min_length=1, max_length=5000, description="Audio script")
    voice_id: str = Field(default="ja-JP-NanamiNeural", description="TTS voice ID")
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Job priority")
    settings: GenerationSettings = Field(default_factory=GenerationSettings)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "1girl, solo, anime, looking at viewer, smile, school uniform",
                "negative_prompt": "lowres, bad anatomy, bad hands",
                "script": "Hello, welcome to my channel!",
                "voice_id": "ja-JP-NanamiNeural",
                "priority": "normal",
                "settings": {
                    "width": 512,
                    "height": 512,
                    "fps": 15,
                    "steps": 20
                }
            }
        }


class JobResponse(BaseModel):
    """Response containing job information."""
    id: str
    prompt: str
    negative_prompt: str
    script: str
    voice_id: str

    width: int
    height: int
    fps: int
    steps: int
    cfg_scale: float
    seed: int
    checkpoint: str
    motion_module: str

    status: JobStatus
    priority: JobPriority
    progress: float
    progress_message: str

    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    output_video: Optional[str]
    output_audio: Optional[str]
    thumbnail: Optional[str]
    error_message: Optional[str]

    audio_duration: float
    frame_count: int
    generation_time: float

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Response containing list of jobs."""
    jobs: List[JobResponse]
    total: int


class QueueStatsResponse(BaseModel):
    """Response containing queue statistics."""
    total_jobs: int
    queue_length: int
    running: int
    max_concurrent: int
    status_counts: Dict[str, int]


class VoiceInfo(BaseModel):
    """Information about an available TTS voice."""
    id: str
    name: str
    gender: str
    locale: str
    style: Optional[str] = None


class VoiceListResponse(BaseModel):
    """Response containing available voices."""
    voices: List[VoiceInfo]


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    file: str
    type: str  # "checkpoint", "motion_module", "lora", "vae"
    description: Optional[str] = None
    size_mb: Optional[float] = None


class ModelListResponse(BaseModel):
    """Response containing available models."""
    checkpoints: List[ModelInfo]
    motion_modules: List[ModelInfo]
    loras: List[ModelInfo]
    vaes: List[ModelInfo]


class SystemStatus(BaseModel):
    """System status information."""
    app_name: str
    version: str
    comfyui_connected: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    queue_stats: QueueStatsResponse


class GeneratePreviewRequest(BaseModel):
    """Request to generate a preview image."""
    prompt: str
    negative_prompt: str = "lowres, bad anatomy"
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    checkpoint: str = "dreamshaper_8.safetensors"


class TTSPreviewRequest(BaseModel):
    """Request to preview TTS audio."""
    text: str = Field(..., min_length=1, max_length=500)
    voice_id: str = "ja-JP-NanamiNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None

