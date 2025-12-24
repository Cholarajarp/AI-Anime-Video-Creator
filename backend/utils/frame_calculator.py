"""
Frame Calculator Utility
========================
Utilities for calculating video frames from audio duration.
"""

import math
from typing import Tuple


def calculate_frame_count(
    audio_duration: float,
    fps: int = 15,
    min_frames: int = 16,
    max_frames: int = 120
) -> int:
    """
    Calculate the number of video frames needed for a given audio duration.

    This is the core of the "Audio-First" pipeline design.

    Args:
        audio_duration: Duration of audio in seconds
        fps: Target frames per second
        min_frames: Minimum number of frames (AnimateDiff context window)
        max_frames: Maximum frames to prevent OOM

    Returns:
        Number of frames to generate
    """
    calculated = math.ceil(audio_duration * fps)
    return max(min_frames, min(calculated, max_frames))


def calculate_video_duration(
    frame_count: int,
    fps: int = 15
) -> float:
    """
    Calculate video duration from frame count.

    Args:
        frame_count: Number of frames
        fps: Frames per second

    Returns:
        Duration in seconds
    """
    return frame_count / fps


def calculate_loop_count(
    video_duration: float,
    audio_duration: float
) -> int:
    """
    Calculate how many times a video needs to loop to match audio.

    Args:
        video_duration: Original video duration in seconds
        audio_duration: Target audio duration in seconds

    Returns:
        Number of complete loops needed
    """
    if video_duration <= 0:
        return 1
    return math.ceil(audio_duration / video_duration)


def estimate_generation_time(
    frame_count: int,
    steps: int = 20,
    width: int = 512,
    height: int = 512,
    base_time_per_step: float = 0.5
) -> float:
    """
    Estimate generation time based on parameters.

    This is a rough estimate for user feedback.

    Args:
        frame_count: Number of frames to generate
        steps: Diffusion steps
        width: Video width
        height: Video height
        base_time_per_step: Base time per step (GPU dependent)

    Returns:
        Estimated time in seconds
    """
    # Scale factor for resolution
    resolution_factor = (width * height) / (512 * 512)

    # Time scales with frames, steps, and resolution
    estimated = frame_count * steps * base_time_per_step * resolution_factor

    # Add overhead for model loading, VAE decode, etc.
    overhead = 10  # seconds

    return estimated + overhead


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def get_resolution_options() -> list[Tuple[str, Tuple[int, int]]]:
    """
    Get list of supported resolution options.

    Returns:
        List of (display_name, (width, height)) tuples
    """
    return [
        ("256x256 (Fast)", (256, 256)),
        ("384x384 (Draft)", (384, 384)),
        ("512x512 (Standard)", (512, 512)),
        ("512x768 (Portrait)", (512, 768)),
        ("768x512 (Landscape)", (768, 512)),
        ("768x768 (High)", (768, 768)),
        ("1024x1024 (Ultra)", (1024, 1024)),
    ]


def get_fps_options() -> list[Tuple[str, int]]:
    """
    Get list of supported FPS options.

    Returns:
        List of (display_name, fps) tuples
    """
    return [
        ("8 FPS (Slow Motion)", 8),
        ("12 FPS (Anime Style)", 12),
        ("15 FPS (Standard)", 15),
        ("24 FPS (Film)", 24),
        ("30 FPS (Smooth)", 30),
    ]


def validate_generation_params(
    width: int,
    height: int,
    fps: int,
    steps: int,
    frame_count: int,
    vram_gb: float = 12.0
) -> Tuple[bool, str]:
    """
    Validate generation parameters against VRAM constraints.

    Args:
        width: Video width
        height: Video height
        fps: Frames per second
        steps: Diffusion steps
        frame_count: Number of frames
        vram_gb: Available VRAM in GB

    Returns:
        Tuple of (is_valid, message)
    """
    # Estimate VRAM usage (rough approximation)
    # Base model: ~4GB
    # Motion module: ~2GB
    # Per-frame latent: ~0.02GB per frame at 512x512

    resolution_factor = (width * height) / (512 * 512)
    frame_vram = frame_count * 0.02 * resolution_factor
    total_estimated = 4 + 2 + frame_vram

    if total_estimated > vram_gb:
        return False, f"Estimated VRAM usage ({total_estimated:.1f}GB) exceeds available ({vram_gb}GB). Reduce resolution or frame count."

    if frame_count > 120:
        return False, "Frame count exceeds maximum (120). Use shorter audio or lower FPS."

    if width > 1024 or height > 1024:
        return False, "Resolution exceeds maximum (1024x1024)."

    if width % 64 != 0 or height % 64 != 0:
        return False, "Width and height must be divisible by 64."

    return True, "Parameters valid"

