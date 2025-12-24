"""
Job Manager - Queue and State Management
=========================================
This module handles job queuing, status tracking, and execution management.
Provides async job processing with priority support and real-time updates.
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from loguru import logger
import json
import aiofiles
from collections import deque


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


@dataclass
class GenerationJob:
    """Represents a video generation job."""
    id: str
    prompt: str
    negative_prompt: str
    script: str
    voice_id: str

    # Generation settings
    width: int = 512
    height: int = 512
    fps: int = 15
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    checkpoint: str = "dreamshaper_8.safetensors"
    motion_module: str = "mm_sd_v15_v2.ckpt"

    # Job metadata
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    progress: float = 0.0
    progress_message: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    output_video: Optional[str] = None
    output_audio: Optional[str] = None
    thumbnail: Optional[str] = None
    error_message: Optional[str] = None

    # Metrics
    audio_duration: float = 0.0
    frame_count: int = 0
    generation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "script": self.script,
            "voice_id": self.voice_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "checkpoint": self.checkpoint,
            "motion_module": self.motion_module,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_video": self.output_video,
            "output_audio": self.output_audio,
            "thumbnail": self.thumbnail,
            "error_message": self.error_message,
            "audio_duration": self.audio_duration,
            "frame_count": self.frame_count,
            "generation_time": self.generation_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationJob":
        """Create job from dictionary."""
        job = cls(
            id=data["id"],
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", ""),
            script=data["script"],
            voice_id=data["voice_id"],
            width=data.get("width", 512),
            height=data.get("height", 512),
            fps=data.get("fps", 15),
            steps=data.get("steps", 20),
            cfg_scale=data.get("cfg_scale", 7.0),
            seed=data.get("seed", -1),
            checkpoint=data.get("checkpoint", "dreamshaper_8.safetensors"),
            motion_module=data.get("motion_module", "mm_sd_v15_v2.ckpt")
        )
        job.status = JobStatus(data.get("status", "pending"))
        job.priority = JobPriority(data.get("priority", "normal"))
        job.progress = data.get("progress", 0.0)
        job.progress_message = data.get("progress_message", "")
        job.output_video = data.get("output_video")
        job.output_audio = data.get("output_audio")
        job.thumbnail = data.get("thumbnail")
        job.error_message = data.get("error_message")
        job.audio_duration = data.get("audio_duration", 0.0)
        job.frame_count = data.get("frame_count", 0)
        job.generation_time = data.get("generation_time", 0.0)

        if data.get("created_at"):
            job.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            job.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            job.completed_at = datetime.fromisoformat(data["completed_at"])

        return job


class JobManager:
    """
    Manages the video generation job queue and execution.

    Features:
    - Priority-based queue management
    - Concurrent job limits
    - Real-time progress callbacks
    - Job persistence to disk
    - Automatic cleanup
    """

    def __init__(
        self,
        max_concurrent: int = 2,
        data_dir: Path = Path("./data"),
        auto_persist: bool = True
    ):
        self.max_concurrent = max_concurrent
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.auto_persist = auto_persist

        # Job storage
        self.jobs: Dict[str, GenerationJob] = {}
        self.queue: deque = deque()

        # Callbacks
        self.progress_callbacks: List[Callable[[str, float, str], Awaitable[None]]] = []
        self.status_callbacks: List[Callable[[str, JobStatus], Awaitable[None]]] = []

        # Processing state
        self.running_jobs: set = set()
        self._processor_task: Optional[asyncio.Task] = None
        self._running = False

        # Load persisted jobs
        self._load_jobs()

    def _get_jobs_file(self) -> Path:
        """Get path to jobs persistence file."""
        return self.data_dir / "jobs.json"

    def _load_jobs(self):
        """Load jobs from disk."""
        jobs_file = self._get_jobs_file()
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    data = json.load(f)
                    for job_data in data.get("jobs", []):
                        job = GenerationJob.from_dict(job_data)
                        self.jobs[job.id] = job
                        # Re-queue pending jobs
                        if job.status in [JobStatus.PENDING, JobStatus.QUEUED]:
                            self.queue.append(job.id)
                logger.info(f"Loaded {len(self.jobs)} jobs from disk")
            except Exception as e:
                logger.error(f"Failed to load jobs: {e}")

    async def _save_jobs(self):
        """Persist jobs to disk."""
        if not self.auto_persist:
            return
        try:
            jobs_data = {
                "jobs": [job.to_dict() for job in self.jobs.values()]
            }
            async with aiofiles.open(self._get_jobs_file(), 'w') as f:
                await f.write(json.dumps(jobs_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")

    def register_progress_callback(
        self,
        callback: Callable[[str, float, str], Awaitable[None]]
    ):
        """Register callback for progress updates."""
        self.progress_callbacks.append(callback)

    def register_status_callback(
        self,
        callback: Callable[[str, JobStatus], Awaitable[None]]
    ):
        """Register callback for status changes."""
        self.status_callbacks.append(callback)

    async def _notify_progress(self, job_id: str, progress: float, message: str):
        """Notify all progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                await callback(job_id, progress, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    async def _notify_status(self, job_id: str, status: JobStatus):
        """Notify all status callbacks."""
        for callback in self.status_callbacks:
            try:
                await callback(job_id, status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def create_job(
        self,
        prompt: str,
        script: str,
        voice_id: str = "ja-JP-NanamiNeural",
        negative_prompt: str = "",
        priority: JobPriority = JobPriority.NORMAL,
        **kwargs
    ) -> GenerationJob:
        """
        Create a new generation job.

        Args:
            prompt: Visual prompt for image generation
            script: Text script for TTS audio
            voice_id: Edge-TTS voice identifier
            negative_prompt: Negative prompt for generation
            priority: Job priority level
            **kwargs: Additional generation settings

        Returns:
            Created GenerationJob instance
        """
        job_id = str(uuid.uuid4())

        # Default negative prompt if not provided
        if not negative_prompt:
            negative_prompt = (
                "lowres, bad anatomy, bad hands, text, error, "
                "missing fingers, extra digit, fewer digits, cropped, "
                "worst quality, low quality, normal quality, jpeg artifacts, "
                "signature, watermark, username, blurry"
            )

        job = GenerationJob(
            id=job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            script=script,
            voice_id=voice_id,
            priority=priority,
            **kwargs
        )

        self.jobs[job_id] = job
        logger.info(f"Created job {job_id}")

        return job

    async def submit_job(self, job: GenerationJob) -> str:
        """
        Submit a job to the queue.

        Returns:
            Job ID
        """
        job.status = JobStatus.QUEUED

        # Priority insertion
        if job.priority == JobPriority.HIGH:
            self.queue.appendleft(job.id)
        else:
            self.queue.append(job.id)

        await self._notify_status(job.id, JobStatus.QUEUED)
        await self._save_jobs()

        logger.info(f"Job {job.id} queued (priority: {job.priority.value})")

        return job.id

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[GenerationJob]:
        """Get all jobs sorted by creation time."""
        return sorted(
            self.jobs.values(),
            key=lambda j: j.created_at,
            reverse=True
        )

    def get_queue(self) -> List[GenerationJob]:
        """Get jobs currently in queue."""
        return [
            self.jobs[jid] for jid in self.queue
            if jid in self.jobs
        ]

    def get_running_jobs(self) -> List[GenerationJob]:
        """Get currently running jobs."""
        return [
            self.jobs[jid] for jid in self.running_jobs
            if jid in self.jobs
        ]

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.

        Returns:
            True if cancelled, False if not found or already completed
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now()

        # Remove from queue if present
        if job_id in self.queue:
            self.queue.remove(job_id)

        await self._notify_status(job_id, JobStatus.CANCELLED)
        await self._save_jobs()

        logger.info(f"Job {job_id} cancelled")
        return True

    async def update_progress(
        self,
        job_id: str,
        progress: float,
        message: str = ""
    ):
        """Update job progress."""
        job = self.jobs.get(job_id)
        if job:
            job.progress = progress
            job.progress_message = message
            await self._notify_progress(job_id, progress, message)

    async def complete_job(
        self,
        job_id: str,
        output_video: str,
        output_audio: str,
        thumbnail: Optional[str] = None
    ):
        """Mark job as completed with outputs."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.output_video = output_video
            job.output_audio = output_audio
            job.thumbnail = thumbnail
            job.progress = 1.0

            if job.started_at:
                job.generation_time = (
                    job.completed_at - job.started_at
                ).total_seconds()

            self.running_jobs.discard(job_id)

            await self._notify_status(job_id, JobStatus.COMPLETED)
            await self._save_jobs()

            logger.info(f"Job {job_id} completed in {job.generation_time:.2f}s")

    async def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed."""
        job = self.jobs.get(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = error_message

            self.running_jobs.discard(job_id)

            await self._notify_status(job_id, JobStatus.FAILED)
            await self._save_jobs()

            logger.error(f"Job {job_id} failed: {error_message}")

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job and its outputs."""
        if job_id in self.jobs:
            job = self.jobs[job_id]

            # Don't delete running jobs
            if job.status == JobStatus.RUNNING:
                return False

            # Remove from queue
            if job_id in self.queue:
                self.queue.remove(job_id)

            del self.jobs[job_id]
            await self._save_jobs()

            logger.info(f"Job {job_id} deleted")
            return True
        return False

    async def cleanup_old_jobs(self, max_age_days: int = 7):
        """Remove old completed/failed jobs."""
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=max_age_days)

        to_delete = []
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.completed_at and job.completed_at < cutoff:
                    to_delete.append(job_id)

        for job_id in to_delete:
            await self.delete_job(job_id)

        logger.info(f"Cleaned up {len(to_delete)} old jobs")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = len([
                j for j in self.jobs.values()
                if j.status == status
            ])

        return {
            "total_jobs": len(self.jobs),
            "queue_length": len(self.queue),
            "running": len(self.running_jobs),
            "max_concurrent": self.max_concurrent,
            "status_counts": status_counts
        }


class JobExecutor:
    """
    Executes generation jobs using the pipeline services.

    This class orchestrates the full generation pipeline:
    1. Generate audio from script
    2. Calculate frame count
    3. Generate video with ComfyUI
    4. Mux audio and video
    """

    def __init__(
        self,
        job_manager: JobManager,
        comfyui_client,
        tts_service,
        ffmpeg_service,
        output_dir: Path = Path("./outputs")
    ):
        self.job_manager = job_manager
        self.comfyui = comfyui_client
        self.tts = tts_service
        self.ffmpeg = ffmpeg_service
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the job processor."""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_queue())
        logger.info("Job executor started")

    async def stop(self):
        """Stop the job processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Job executor stopped")

    async def _process_queue(self):
        """Main queue processing loop."""
        while self._running:
            try:
                # Check if we can start new jobs
                if (
                    len(self.job_manager.running_jobs) < self.job_manager.max_concurrent
                    and self.job_manager.queue
                ):
                    job_id = self.job_manager.queue.popleft()
                    job = self.job_manager.get_job(job_id)

                    if job and job.status == JobStatus.QUEUED:
                        asyncio.create_task(self._execute_job(job))

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(5)

    async def _execute_job(self, job: GenerationJob):
        """Execute a single generation job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self.job_manager.running_jobs.add(job.id)

        await self.job_manager._notify_status(job.id, JobStatus.RUNNING)

        try:
            # Step 1: Generate audio
            await self.job_manager.update_progress(
                job.id, 0.1, "Generating audio..."
            )

            audio_result = await self.tts.generate_audio(
                text=job.script,
                voice_id=job.voice_id,
                output_filename=f"{job.id}_audio.mp3"
            )

            if not audio_result.success:
                raise Exception(f"Audio generation failed: {audio_result.error_message}")

            job.audio_duration = audio_result.duration_seconds
            job.output_audio = str(audio_result.file_path)

            # Step 2: Calculate frames
            job.frame_count = self.tts.calculate_frame_count(
                audio_result.duration_seconds,
                job.fps
            )

            await self.job_manager.update_progress(
                job.id, 0.2,
                f"Audio ready ({audio_result.duration_seconds:.1f}s), generating {job.frame_count} frames..."
            )

            # Step 3: Generate video
            from backend.services.comfyui_client import WorkflowBuilder

            workflow = WorkflowBuilder.create_animatediff_workflow(
                prompt=f"masterpiece, best quality, {job.prompt}",
                negative_prompt=job.negative_prompt,
                checkpoint=job.checkpoint,
                motion_module=job.motion_module,
                frame_count=job.frame_count,
                width=job.width,
                height=job.height,
                steps=job.steps,
                cfg=job.cfg_scale,
                fps=job.fps,
                seed=job.seed
            )

            def progress_callback(msg: str, prog: float):
                asyncio.create_task(
                    self.job_manager.update_progress(
                        job.id,
                        0.2 + prog * 0.6,
                        msg
                    )
                )

            video_result = await self.comfyui.queue_prompt(
                workflow,
                progress_callback=progress_callback
            )

            if not video_result.success:
                raise Exception(f"Video generation failed: {video_result.error_message}")

            # Step 4: Download generated video
            await self.job_manager.update_progress(
                job.id, 0.85, "Downloading video..."
            )

            if video_result.output_files:
                video_filename = video_result.output_files[0]
                video_path = await self.comfyui.download_output(
                    video_filename,
                    self.output_dir
                )
            else:
                raise Exception("No video output generated")

            # Step 5: Mux audio and video
            await self.job_manager.update_progress(
                job.id, 0.9, "Combining audio and video..."
            )

            final_output = f"{job.id}_final.mp4"
            mux_result = self.ffmpeg.mux_audio_video(
                video_path=video_path,
                audio_path=audio_result.file_path,
                output_filename=final_output,
                loop_video=True
            )

            if not mux_result.success:
                raise Exception(f"Muxing failed: {mux_result.error_message}")

            # Step 6: Generate thumbnail
            thumbnail_path = self.ffmpeg.extract_thumbnail(
                video_path=mux_result.output_path,
                output_filename=f"{job.id}_thumb.jpg"
            )

            # Complete job
            await self.job_manager.complete_job(
                job.id,
                output_video=str(mux_result.output_path),
                output_audio=str(audio_result.file_path),
                thumbnail=str(thumbnail_path) if thumbnail_path else None
            )

        except Exception as e:
            logger.exception(f"Job {job.id} execution failed")
            await self.job_manager.fail_job(job.id, str(e))

