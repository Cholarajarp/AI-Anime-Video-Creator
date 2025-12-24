"""
Eye Blink + Breathing Motion Engine
Add life-like idle motion to templates
"""

import math
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class FrameTransform:
    """Per-frame transform data"""
    frame: int
    scale: float = 1.0
    eye_state: str = "open"
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "frame": self.frame,
            "scale": self.scale,
            "eye_state": self.eye_state,
            "offset_x": self.offset_x,
            "offset_y": self.offset_y,
            "rotation": self.rotation
        }


@dataclass
class IdleAnimationConfig:
    """Configuration for idle animation"""
    # Blinking
    blink_interval_min: float = 2.0  # seconds
    blink_interval_max: float = 6.0  # seconds
    blink_duration_frames: int = 4

    # Breathing
    breathing_enabled: bool = True
    breathing_scale_amplitude: float = 0.02  # ±2%
    breathing_cycle_duration: float = 3.0  # seconds

    # Micro movements
    micro_movement_enabled: bool = True
    micro_movement_amplitude: float = 2.0  # pixels
    micro_movement_frequency: float = 0.5  # Hz

    # Camera shake
    camera_shake_enabled: bool = False
    camera_shake_amplitude: float = 1.0  # pixels
    camera_shake_frequency: float = 2.0  # Hz

    # Hair/cloth sway
    sway_enabled: bool = True
    sway_amplitude: float = 3.0  # pixels
    sway_frequency: float = 0.3  # Hz


class IdleAnimationEngine:
    """Generate procedural idle animations"""

    def __init__(self, config: Optional[IdleAnimationConfig] = None, seed: int = 42):
        self.config = config or IdleAnimationConfig()
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_idle_animation(
        self,
        frame_count: int,
        fps: int = 24
    ) -> Dict[int, Dict]:
        """
        Generate per-frame transform map for idle animation

        Args:
            frame_count: Total number of frames
            fps: Frames per second

        Returns:
            Dictionary mapping frame index to transform data
        """
        self._rng = random.Random(self.seed)  # Reset for determinism

        # Pre-calculate all animation tracks
        blink_track = self._generate_blink_track(frame_count, fps)
        breathing_track = self._generate_breathing_track(frame_count, fps)
        micro_movement_track = self._generate_micro_movement_track(frame_count, fps)
        camera_shake_track = self._generate_camera_shake_track(frame_count, fps)
        sway_track = self._generate_sway_track(frame_count, fps)

        # Combine all tracks
        animation_map = {}

        for frame in range(frame_count):
            transform = FrameTransform(frame=frame)

            # Apply blink state
            transform.eye_state = blink_track[frame]

            # Apply breathing scale
            if self.config.breathing_enabled:
                transform.scale = breathing_track[frame]

            # Apply micro movements
            if self.config.micro_movement_enabled:
                transform.offset_x += micro_movement_track[frame][0]
                transform.offset_y += micro_movement_track[frame][1]

            # Apply camera shake
            if self.config.camera_shake_enabled:
                transform.offset_x += camera_shake_track[frame][0]
                transform.offset_y += camera_shake_track[frame][1]

            # Apply sway
            if self.config.sway_enabled:
                transform.offset_x += sway_track[frame][0]
                transform.offset_y += sway_track[frame][1]

            animation_map[frame] = transform.to_dict()

        logger.info(f"✅ Generated idle animation: {frame_count} frames")

        return animation_map

    def _generate_blink_track(self, frame_count: int, fps: int) -> List[str]:
        """Generate eye blink states"""
        track = ["open"] * frame_count

        current_frame = 0

        while current_frame < frame_count:
            # Random interval until next blink
            interval_seconds = self._rng.uniform(
                self.config.blink_interval_min,
                self.config.blink_interval_max
            )
            interval_frames = int(interval_seconds * fps)

            blink_start = current_frame + interval_frames

            if blink_start >= frame_count:
                break

            # Apply blink (close-open sequence)
            blink_duration = self.config.blink_duration_frames

            for i in range(blink_duration):
                frame_idx = blink_start + i
                if frame_idx < frame_count:
                    # First half: closing, second half: opening
                    if i < blink_duration // 2:
                        track[frame_idx] = "closing"
                    elif i == blink_duration // 2:
                        track[frame_idx] = "closed"
                    else:
                        track[frame_idx] = "opening"

            current_frame = blink_start + blink_duration

        return track

    def _generate_breathing_track(self, frame_count: int, fps: int) -> List[float]:
        """Generate breathing scale oscillation"""
        track = []

        cycle_frames = int(self.config.breathing_cycle_duration * fps)
        amplitude = self.config.breathing_scale_amplitude

        for frame in range(frame_count):
            # Sinusoidal breathing pattern
            phase = (frame % cycle_frames) / cycle_frames * 2 * math.pi
            scale = 1.0 + amplitude * math.sin(phase)
            track.append(scale)

        return track

    def _generate_micro_movement_track(
        self,
        frame_count: int,
        fps: int
    ) -> List[tuple]:
        """Generate subtle micro movements"""
        track = []

        amplitude = self.config.micro_movement_amplitude
        freq = self.config.micro_movement_frequency

        # Use Perlin-like smooth noise
        phase_x = self._rng.uniform(0, 100)
        phase_y = self._rng.uniform(0, 100)

        for frame in range(frame_count):
            t = frame / fps

            # Multiple sine waves for organic movement
            offset_x = amplitude * (
                0.5 * math.sin(2 * math.pi * freq * t + phase_x) +
                0.3 * math.sin(2 * math.pi * freq * 1.7 * t + phase_x * 1.3) +
                0.2 * math.sin(2 * math.pi * freq * 2.3 * t + phase_x * 0.7)
            )

            offset_y = amplitude * (
                0.5 * math.sin(2 * math.pi * freq * t + phase_y) +
                0.3 * math.sin(2 * math.pi * freq * 1.5 * t + phase_y * 1.2) +
                0.2 * math.sin(2 * math.pi * freq * 2.1 * t + phase_y * 0.8)
            )

            track.append((offset_x, offset_y))

        return track

    def _generate_camera_shake_track(
        self,
        frame_count: int,
        fps: int
    ) -> List[tuple]:
        """Generate camera shake (optional)"""
        if not self.config.camera_shake_enabled:
            return [(0.0, 0.0)] * frame_count

        track = []
        amplitude = self.config.camera_shake_amplitude
        freq = self.config.camera_shake_frequency

        for frame in range(frame_count):
            t = frame / fps

            # High-frequency noise for shake
            shake_x = amplitude * (
                self._rng.uniform(-1, 1) * 0.3 +
                math.sin(2 * math.pi * freq * t) * 0.7
            )
            shake_y = amplitude * (
                self._rng.uniform(-1, 1) * 0.3 +
                math.cos(2 * math.pi * freq * t) * 0.7
            )

            track.append((shake_x, shake_y))

        return track

    def _generate_sway_track(
        self,
        frame_count: int,
        fps: int
    ) -> List[tuple]:
        """Generate hair/cloth sway motion"""
        if not self.config.sway_enabled:
            return [(0.0, 0.0)] * frame_count

        track = []
        amplitude = self.config.sway_amplitude
        freq = self.config.sway_frequency

        phase = self._rng.uniform(0, 2 * math.pi)

        for frame in range(frame_count):
            t = frame / fps

            # Gentle swaying motion
            sway_x = amplitude * math.sin(2 * math.pi * freq * t + phase)
            sway_y = amplitude * 0.3 * math.sin(2 * math.pi * freq * 2 * t + phase)

            track.append((sway_x, sway_y))

        return track

    def get_transform_at_frame(
        self,
        animation_map: Dict[int, Dict],
        frame: int
    ) -> Dict:
        """Get transform data for a specific frame with interpolation"""
        if frame in animation_map:
            return animation_map[frame]

        # Find nearest frames for interpolation
        frames = sorted(animation_map.keys())
        if frame < frames[0]:
            return animation_map[frames[0]]
        if frame > frames[-1]:
            return animation_map[frames[-1]]

        # Linear interpolation
        for i, f in enumerate(frames):
            if f > frame:
                prev_frame = frames[i - 1]
                next_frame = f
                t = (frame - prev_frame) / (next_frame - prev_frame)

                prev_data = animation_map[prev_frame]
                next_data = animation_map[next_frame]

                return {
                    "frame": frame,
                    "scale": prev_data["scale"] + t * (next_data["scale"] - prev_data["scale"]),
                    "eye_state": next_data["eye_state"],
                    "offset_x": prev_data["offset_x"] + t * (next_data["offset_x"] - prev_data["offset_x"]),
                    "offset_y": prev_data["offset_y"] + t * (next_data["offset_y"] - prev_data["offset_y"]),
                    "rotation": prev_data.get("rotation", 0) + t * (next_data.get("rotation", 0) - prev_data.get("rotation", 0))
                }

        return animation_map[frames[-1]]


def generate_idle_animation(
    frame_count: int,
    fps: int = 24,
    seed: int = 42
) -> Dict[int, Dict]:
    """Convenience function for idle animation generation"""
    engine = IdleAnimationEngine(seed=seed)
    return engine.generate_idle_animation(frame_count, fps)


def generate_combined_animation(
    frame_count: int,
    fps: int = 24,
    audio_path: Optional[str] = None,
    seed: int = 42
) -> Dict[int, Dict]:
    """Generate combined idle + mouth sync animation"""
    # Generate idle animation
    idle_map = generate_idle_animation(frame_count, fps, seed)

    # Generate mouth sync if audio provided
    if audio_path:
        from .mouth_sync import generate_mouth_animation
        mouth_map, confidence = generate_mouth_animation(audio_path, fps)

        # Merge mouth states into idle animation
        for mouth_data in mouth_map:
            frame = mouth_data["frame"]
            if frame in idle_map:
                idle_map[frame]["mouth"] = mouth_data["mouth"]
                idle_map[frame]["mouth_intensity"] = mouth_data["intensity"]

    return idle_map

