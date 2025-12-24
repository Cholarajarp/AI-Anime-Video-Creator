"""
REST API Routes
===============
FastAPI routes for the video generation API.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional
import os

from backend.models.job import (
    CreateJobRequest,
    JobResponse,
    JobListResponse,
    QueueStatsResponse,
    VoiceListResponse,
    VoiceInfo,
    ModelListResponse,
    ModelInfo,
    SystemStatus,
    GeneratePreviewRequest,
    TTSPreviewRequest,
    ErrorResponse
)
from backend.models.settings import get_settings
from backend.services.job_manager import JobManager, GenerationJob, JobStatus, JobPriority
from backend.services.tts_service import TTSService
from backend.services.comfyui_client import ComfyUIClient
from backend.services.ffmpeg_service import FFmpegService

# Create router
router = APIRouter(prefix="/api", tags=["API"])

# Service instances (initialized on startup)
job_manager: Optional[JobManager] = None
tts_service: Optional[TTSService] = None
comfyui_client: Optional[ComfyUIClient] = None
ffmpeg_service: Optional[FFmpegService] = None


def get_job_manager() -> JobManager:
    """Dependency to get job manager."""
    if job_manager is None:
        raise HTTPException(500, "Job manager not initialized")
    return job_manager


def get_tts_service() -> TTSService:
    """Dependency to get TTS service."""
    if tts_service is None:
        raise HTTPException(500, "TTS service not initialized")
    return tts_service


def get_comfyui_client() -> ComfyUIClient:
    """Dependency to get ComfyUI client."""
    if comfyui_client is None:
        raise HTTPException(500, "ComfyUI client not initialized")
    return comfyui_client


# =============================================================================
# Job Management Endpoints
# =============================================================================

@router.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
    jm: JobManager = Depends(get_job_manager)
):
    """Create a new video generation job."""
    job = jm.create_job(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        script=request.script,
        voice_id=request.voice_id,
        priority=JobPriority(request.priority.value),
        width=request.settings.width,
        height=request.settings.height,
        fps=request.settings.fps,
        steps=request.settings.steps,
        cfg_scale=request.settings.cfg_scale,
        seed=request.settings.seed,
        checkpoint=request.settings.checkpoint,
        motion_module=request.settings.motion_module
    )

    # Queue the job
    await jm.submit_job(job)

    return _job_to_response(job)


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    jm: JobManager = Depends(get_job_manager)
):
    """List all jobs with optional filtering."""
    jobs = jm.get_all_jobs()

    # Filter by status if provided
    if status:
        try:
            status_filter = JobStatus(status)
            jobs = [j for j in jobs if j.status == status_filter]
        except ValueError:
            pass

    # Paginate
    total = len(jobs)
    jobs = jobs[offset:offset + limit]

    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    jm: JobManager = Depends(get_job_manager)
):
    """Get a specific job by ID."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return _job_to_response(job)


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str,
    jm: JobManager = Depends(get_job_manager)
):
    """Delete a job."""
    success = await jm.delete_job(job_id)
    if not success:
        raise HTTPException(404, f"Job {job_id} not found or cannot be deleted")
    return {"status": "deleted", "job_id": job_id}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    jm: JobManager = Depends(get_job_manager)
):
    """Cancel a pending or running job."""
    success = await jm.cancel_job(job_id)
    if not success:
        raise HTTPException(400, f"Job {job_id} cannot be cancelled")
    return {"status": "cancelled", "job_id": job_id}


@router.get("/jobs/{job_id}/video")
async def get_job_video(
    job_id: str,
    jm: JobManager = Depends(get_job_manager)
):
    """Download the generated video for a job."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if not job.output_video:
        raise HTTPException(400, "Video not yet available")

    video_path = Path(job.output_video)
    if not video_path.exists():
        raise HTTPException(404, "Video file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )


@router.get("/jobs/{job_id}/audio")
async def get_job_audio(
    job_id: str,
    jm: JobManager = Depends(get_job_manager)
):
    """Download the generated audio for a job."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if not job.output_audio:
        raise HTTPException(400, "Audio not yet available")

    audio_path = Path(job.output_audio)
    if not audio_path.exists():
        raise HTTPException(404, "Audio file not found")

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=f"{job_id}.mp3"
    )


@router.get("/jobs/{job_id}/thumbnail")
async def get_job_thumbnail(
    job_id: str,
    jm: JobManager = Depends(get_job_manager)
):
    """Get the thumbnail for a job."""
    job = jm.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    if not job.thumbnail:
        raise HTTPException(400, "Thumbnail not available")

    thumb_path = Path(job.thumbnail)
    if not thumb_path.exists():
        raise HTTPException(404, "Thumbnail file not found")

    return FileResponse(
        thumb_path,
        media_type="image/jpeg",
        filename=f"{job_id}_thumb.jpg"
    )


# =============================================================================
# Queue Endpoints
# =============================================================================

@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats(jm: JobManager = Depends(get_job_manager)):
    """Get queue statistics."""
    stats = jm.get_stats()
    return QueueStatsResponse(**stats)


@router.get("/queue", response_model=JobListResponse)
async def get_queue(jm: JobManager = Depends(get_job_manager)):
    """Get jobs currently in queue."""
    jobs = jm.get_queue()
    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=len(jobs)
    )


@router.get("/queue/running", response_model=JobListResponse)
async def get_running_jobs(jm: JobManager = Depends(get_job_manager)):
    """Get currently running jobs."""
    jobs = jm.get_running_jobs()
    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=len(jobs)
    )


# =============================================================================
# TTS Endpoints
# =============================================================================

@router.get("/voices", response_model=VoiceListResponse)
async def list_voices():
    """List available TTS voices."""
    voices = TTSService.get_available_voices()
    return VoiceListResponse(
        voices=[
            VoiceInfo(
                id=v.id,
                name=v.name,
                gender=v.gender,
                locale=v.locale,
                style=v.style
            )
            for v in voices
        ]
    )


@router.post("/tts/preview")
async def preview_tts(
    request: TTSPreviewRequest,
    tts: TTSService = Depends(get_tts_service)
):
    """Generate a preview of TTS audio."""
    result = await tts.generate_audio(
        text=request.text,
        voice_id=request.voice_id,
        rate=request.rate,
        pitch=request.pitch
    )

    if not result.success:
        raise HTTPException(500, f"TTS failed: {result.error_message}")

    return FileResponse(
        result.file_path,
        media_type="audio/mpeg",
        filename="preview.mp3"
    )


# =============================================================================
# Model Management Endpoints
# =============================================================================

@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List available models."""
    settings = get_settings()

    def scan_models(dir_path: Path, model_type: str) -> List[ModelInfo]:
        models = []
        if dir_path.exists():
            for f in dir_path.glob("*.safetensors"):
                models.append(ModelInfo(
                    name=f.stem,
                    file=f.name,
                    type=model_type,
                    size_mb=f.stat().st_size / (1024 * 1024)
                ))
            for f in dir_path.glob("*.ckpt"):
                models.append(ModelInfo(
                    name=f.stem,
                    file=f.name,
                    type=model_type,
                    size_mb=f.stat().st_size / (1024 * 1024)
                ))
        return models

    return ModelListResponse(
        checkpoints=scan_models(settings.models_dir / "checkpoints", "checkpoint"),
        motion_modules=scan_models(settings.models_dir / "motion_modules", "motion_module"),
        loras=scan_models(settings.models_dir / "loras", "lora"),
        vaes=scan_models(settings.models_dir / "vae", "vae")
    )


# =============================================================================
# System Status Endpoints
# =============================================================================

@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    jm: JobManager = Depends(get_job_manager),
    comfy: ComfyUIClient = Depends(get_comfyui_client)
):
    """Get system status."""
    settings = get_settings()

    # Check ComfyUI connection
    comfyui_connected = await comfy.check_connection()

    # Get GPU info if available
    gpu_name = None
    vram_total = None
    vram_used = None
    gpu_available = False

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
    except Exception:
        pass

    queue_stats = jm.get_stats()

    return SystemStatus(
        app_name=settings.app_name,
        version=settings.app_version,
        comfyui_connected=comfyui_connected,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_used_gb=vram_used,
        queue_stats=QueueStatsResponse(**queue_stats)
    )


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}


# =============================================================================
# Helper Functions
# =============================================================================

def _job_to_response(job: GenerationJob) -> JobResponse:
    """Convert GenerationJob to JobResponse."""
    return JobResponse(
        id=job.id,
        prompt=job.prompt,
        negative_prompt=job.negative_prompt,
        script=job.script,
        voice_id=job.voice_id,
        width=job.width,
        height=job.height,
        fps=job.fps,
        steps=job.steps,
        cfg_scale=job.cfg_scale,
        seed=job.seed,
        checkpoint=job.checkpoint,
        motion_module=job.motion_module,
        status=job.status,
        priority=job.priority,
        progress=job.progress,
        progress_message=job.progress_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        output_video=job.output_video,
        output_audio=job.output_audio,
        thumbnail=job.thumbnail,
        error_message=job.error_message,
        audio_duration=job.audio_duration,
        frame_count=job.frame_count,
        generation_time=job.generation_time
    )


def init_services(
    _job_manager: JobManager,
    _tts_service: TTSService,
    _comfyui_client: ComfyUIClient,
    _ffmpeg_service: FFmpegService
):
    """Initialize service instances for dependency injection."""
    global job_manager, tts_service, comfyui_client, ffmpeg_service
    job_manager = _job_manager
    tts_service = _tts_service
    comfyui_client = _comfyui_client
    ffmpeg_service = _ffmpeg_service

