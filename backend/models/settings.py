"""
Application Settings
====================
Pydantic settings for environment-based configuration.
"""

from pathlib import Path
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = Field(default="Anime Video Creator")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)
    secret_key: str = Field(default="change-this-in-production")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7860)
    workers: int = Field(default=1)

    # ComfyUI
    comfyui_host: str = Field(default="127.0.0.1")
    comfyui_port: int = Field(default=8188)
    comfyui_protocol: str = Field(default="http")
    comfyui_ws_protocol: str = Field(default="ws")
    comfyui_timeout: int = Field(default=600)

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./data/jobs.db")

    # Redis
    redis_url: Optional[str] = Field(default=None)

    # FFmpeg
    ffmpeg_path: str = Field(default="ffmpeg")
    ffprobe_path: str = Field(default="ffprobe")

    # Directories
    output_dir: Path = Field(default=Path("./outputs"))
    temp_dir: Path = Field(default=Path("./temp"))
    models_dir: Path = Field(default=Path("./models"))
    data_dir: Path = Field(default=Path("./data"))
    logs_dir: Path = Field(default=Path("./logs"))

    # Video defaults
    default_fps: int = Field(default=15)
    default_width: int = Field(default=512)
    default_height: int = Field(default=512)
    max_frames: int = Field(default=120)
    default_steps: int = Field(default=20)
    default_cfg_scale: float = Field(default=7.0)

    # Audio
    default_voice: str = Field(default="ja-JP-NanamiNeural")
    audio_format: str = Field(default="mp3")
    audio_sample_rate: int = Field(default=22050)

    # Queue
    max_concurrent_jobs: int = Field(default=2)
    job_timeout_seconds: int = Field(default=600)

    # GPU
    cuda_visible_devices: str = Field(default="0")
    torch_device: str = Field(default="cuda")

    # Logging
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=None)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_directories(self):
        """Create all required directories."""
        for dir_path in [
            self.output_dir,
            self.temp_dir,
            self.models_dir,
            self.data_dir,
            self.logs_dir,
            self.models_dir / "checkpoints",
            self.models_dir / "motion_modules",
            self.models_dir / "loras",
            self.models_dir / "vae"
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @property
    def comfyui_url(self) -> str:
        """Get full ComfyUI URL."""
        return f"{self.comfyui_protocol}://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def comfyui_ws_url(self) -> str:
        """Get ComfyUI WebSocket URL."""
        return f"{self.comfyui_ws_protocol}://{self.comfyui_host}:{self.comfyui_port}/ws"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings

