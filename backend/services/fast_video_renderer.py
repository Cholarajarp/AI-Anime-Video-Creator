"""
Fast Video Renderer - PNG → Video Without Diffusion
Ultra-fast animation using FFmpeg + PIL/OpenCV only
No diffusion models - pure procedural animation
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from loguru import logger

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available, using PIL fallback")

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None
    logger.warning("imageio_ffmpeg not found, video encoding may fail")


class FastVideoRenderer:
    """
    Real-time video rendering from PNG templates
    No diffusion models - pure procedural animation
    """

    def __init__(
        self,
        output_resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        codec: str = "libx264",
        quality: int = 23  # CRF value, lower = better quality
    ):
        self.output_resolution = output_resolution
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """Find FFmpeg executable"""
        # Check common locations
        locations = [
            "ffmpeg",
            "ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg"
        ]

        for loc in locations:
            if shutil.which(loc):
                return loc

        # Try imageio-ffmpeg
        if imageio_ffmpeg is not None:
            try:
                return imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass

        logger.warning("FFmpeg not found, video encoding may fail")
        return "ffmpeg"

    def render_video(
        self,
        template_dir: str,
        output_path: str,
        duration: float = 5.0,
        audio_path: Optional[str] = None,
        animation_config: Optional[Dict] = None
    ) -> bool:
        """
        Render video from template layers

        Args:
            template_dir: Path to template pack
            output_path: Output video file path
            duration: Video duration in seconds
            audio_path: Optional audio file to sync
            animation_config: Animation parameters

        Returns:
            True if successful
        """
        try:
            # Load template layers
            layers = self._load_layers(template_dir)
            if not layers:
                logger.warning("No template layers found, using generated frames")
                layers = self._create_default_layers()

            frame_count = int(duration * self.fps)

            # Generate animation maps
            from .idle_animation import generate_idle_animation
            from .mouth_sync import generate_mouth_animation

            idle_map = generate_idle_animation(frame_count, self.fps)

            mouth_map = {}
            if audio_path and Path(audio_path).exists():
                mouth_frames, _ = generate_mouth_animation(audio_path, self.fps)
                mouth_map = {f["frame"]: f for f in mouth_frames}

            # Render frames
            with tempfile.TemporaryDirectory() as temp_dir:
                frame_paths = self._render_frames(
                    layers,
                    frame_count,
                    idle_map,
                    mouth_map,
                    temp_dir
                )

                if not frame_paths:
                    logger.error("Failed to render frames")
                    return False

                # Encode video with FFmpeg
                success = self._encode_video(
                    frame_paths,
                    output_path,
                    audio_path
                )

                return success

        except Exception as e:
            logger.error(f"Video rendering failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _load_layers(self, template_dir: str) -> Dict[str, np.ndarray]:
        """Load all template layers as numpy arrays"""

        template_path = Path(template_dir)
        if not template_path.exists():
            return {}

        layers = {}

        layer_files = {
            "background": ["background.png", "bg.png"],
            "character": ["character.png", "char.png", "body.png"],
            "mouth_open": ["mouth_open.png", "mouth_o.png"],
            "mouth_closed": ["mouth_closed.png", "mouth_c.png"],
            "eyes_open": ["eyes_open.png", "eye_open.png"],
            "eyes_closed": ["eyes_closed.png", "eye_closed.png"],
        }

        for layer_name, possible_files in layer_files.items():
            for filename in possible_files:
                filepath = template_path / filename
                if filepath.exists():
                    img = Image.open(filepath)
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    img = img.resize(self.output_resolution, Image.Resampling.LANCZOS)
                    layers[layer_name] = np.array(img)
                    logger.debug(f"Loaded layer: {layer_name}")
                    break

        return layers

    def _create_default_layers(self) -> Dict[str, np.ndarray]:
        """Create default layers when no template found"""
        w, h = self.output_resolution

        # Create gradient background
        bg = np.zeros((h, w, 4), dtype=np.uint8)
        for y in range(h):
            t = y / h
            bg[y, :, 0] = int(50 + t * 30)  # R
            bg[y, :, 1] = int(70 + t * 40)  # G
            bg[y, :, 2] = int(130 + t * 50)  # B
            bg[y, :, 3] = 255  # A

        return {"background": bg}

    def _render_frames(
        self,
        layers: Dict[str, np.ndarray],
        frame_count: int,
        idle_map: Dict[int, Dict],
        mouth_map: Dict[int, Dict],
        temp_dir: str
    ) -> List[str]:
        """Render all frames to temporary files"""

        frame_paths = []

        for frame_idx in range(frame_count):
            # Get animation state
            idle_state = idle_map.get(frame_idx, {})
            mouth_state = mouth_map.get(frame_idx, {})

            # Composite frame
            frame = self._composite_frame(
                layers,
                idle_state,
                mouth_state
            )

            # Save frame
            frame_path = Path(temp_dir) / f"frame_{frame_idx:06d}.png"
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(str(frame_path))

            # Progress logging
            if frame_idx % max(1, frame_count // 10) == 0:
                logger.info(f"   Rendered frame {frame_idx + 1}/{frame_count}")

        return frame_paths

    def _composite_frame(
        self,
        layers: Dict[str, np.ndarray],
        idle_state: Dict,
        mouth_state: Dict
    ) -> np.ndarray:
        """Composite layers into single frame"""

        w, h = self.output_resolution

        # Start with background
        if "background" in layers:
            result = layers["background"].copy()
        else:
            result = np.zeros((h, w, 4), dtype=np.uint8)
            result[:, :, 3] = 255

        # Apply transforms from idle animation
        scale = idle_state.get("scale", 1.0)
        offset_x = idle_state.get("offset_x", 0)
        offset_y = idle_state.get("offset_y", 0)
        eye_state = idle_state.get("eye_state", "open")

        # Add character with transform
        if "character" in layers:
            char = self._apply_transform(
                layers["character"],
                scale=scale,
                offset=(offset_x, offset_y)
            )
            result = self._alpha_composite(result, char)

        # Add eyes based on state
        if eye_state in ["closed", "closing"] and "eyes_closed" in layers:
            eyes = self._apply_transform(
                layers["eyes_closed"],
                scale=scale,
                offset=(offset_x, offset_y)
            )
            result = self._alpha_composite(result, eyes)
        elif "eyes_open" in layers:
            eyes = self._apply_transform(
                layers["eyes_open"],
                scale=scale,
                offset=(offset_x, offset_y)
            )
            result = self._alpha_composite(result, eyes)

        # Add mouth based on state
        mouth_open = mouth_state.get("mouth", "closed") == "open"
        if mouth_open and "mouth_open" in layers:
            mouth = self._apply_transform(
                layers["mouth_open"],
                scale=scale,
                offset=(offset_x, offset_y)
            )
            result = self._alpha_composite(result, mouth)
        elif "mouth_closed" in layers:
            mouth = self._apply_transform(
                layers["mouth_closed"],
                scale=scale,
                offset=(offset_x, offset_y)
            )
            result = self._alpha_composite(result, mouth)

        # Convert to RGB for video
        return result[:, :, :3]

    def _apply_transform(
        self,
        layer: np.ndarray,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        rotation: float = 0
    ) -> np.ndarray:
        """Apply spatial transform to layer"""

        if scale == 1.0 and offset == (0, 0) and rotation == 0:
            return layer

        img = Image.fromarray(layer)
        w, h = img.size

        # Apply scale
        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Pad or crop to original size
            result = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            paste_x = (w - new_w) // 2
            paste_y = (h - new_h) // 2
            result.paste(img, (paste_x, paste_y))
            img = result

        # Apply rotation
        if rotation != 0:
            img = img.rotate(rotation, resample=Image.Resampling.BILINEAR, expand=False)

        # Apply offset
        if offset != (0, 0):
            result = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            result.paste(img, (int(offset[0]), int(offset[1])))
            img = result

        return np.array(img)

    def _alpha_composite(
        self,
        base: np.ndarray,
        overlay: np.ndarray
    ) -> np.ndarray:
        """Alpha composite two RGBA images"""

        if overlay.shape[2] < 4:
            # No alpha channel, just copy
            return overlay

        # Normalize alpha
        alpha = overlay[:, :, 3:4].astype(float) / 255.0

        # Composite
        result = base.copy().astype(float)
        result[:, :, :3] = (
            result[:, :, :3] * (1 - alpha) +
            overlay[:, :, :3].astype(float) * alpha
        )

        return result.astype(np.uint8)

    def _encode_video(
        self,
        frame_paths: List[str],
        output_path: str,
        audio_path: Optional[str] = None
    ) -> bool:
        """Encode frames to video using FFmpeg"""

        try:
            # Get frame pattern
            first_frame = frame_paths[0]
            frame_pattern = str(Path(first_frame).parent / "frame_%06d.png")

            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                "-y",  # Overwrite output
                "-framerate", str(self.fps),
                "-i", frame_pattern,
            ]

            # Add audio if provided
            if audio_path and Path(audio_path).exists():
                cmd.extend(["-i", audio_path])

            # Output settings
            cmd.extend([
                "-c:v", self.codec,
                "-crf", str(self.quality),
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
            ])

            # Audio settings
            if audio_path and Path(audio_path).exists():
                cmd.extend([
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest"  # End when shortest stream ends
                ])

            cmd.append(output_path)

            # Run FFmpeg
            logger.info(f"Encoding video: {output_path}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

            logger.info(f"✅ Video encoded: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Video encoding failed: {e}")
            return False

    def render_vtuber_style(
        self,
        template_dir: str,
        output_path: str,
        audio_path: str,
        character_name: str = "Character"
    ) -> bool:
        """
        Render VTuber-style talking avatar video

        Args:
            template_dir: Path to template pack
            output_path: Output video file
            audio_path: Audio file to sync
            character_name: Character name for display

        Returns:
            True if successful
        """

        # Calculate duration from audio
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0
        except ImportError:
            duration = None

        return self.render_video(
            template_dir=template_dir,
            output_path=output_path,
            duration=duration,
            audio_path=audio_path,
            animation_config={
                "mode": "vtuber",
                "character_name": character_name
            }
        )


class SimpleVideoRenderer:
    """Simplified video renderer without external dependencies"""

    def __init__(self, fps: int = 30, resolution: Tuple[int, int] = (512, 512)):
        self.fps = fps
        self.resolution = resolution

    def create_animated_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        audio_path: Optional[str] = None
    ) -> bool:
        """Create video from numpy frames"""
        try:
            import imageio

            # Create video
            writer = imageio.get_writer(
                output_path,
                fps=self.fps,
                codec='libx264',
                quality=8
            )

            for frame in frames:
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                writer.append_data(frame)

            writer.close()

            # Add audio if provided
            if audio_path and Path(audio_path).exists():
                return self._add_audio_to_video(output_path, audio_path)

            return True

        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return False

    def _add_audio_to_video(
        self,
        video_path: str,
        audio_path: str
    ) -> bool:
        """Add audio to existing video"""
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

            temp_output = video_path + ".temp.mp4"

            cmd = [
                ffmpeg_exe,
                "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                temp_output
            ]

            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0:
                os.replace(temp_output, video_path)
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to add audio: {e}")
            return False


def render_fast_video(
    template_dir: str,
    output_path: str,
    duration: float = 5.0,
    audio_path: Optional[str] = None,
    resolution: Tuple[int, int] = (1920, 1080),
    fps: int = 30
) -> bool:
    """Convenience function for fast video rendering"""
    renderer = FastVideoRenderer(
        output_resolution=resolution,
        fps=fps
    )
    return renderer.render_video(
        template_dir=template_dir,
        output_path=output_path,
        duration=duration,
        audio_path=audio_path
    )

