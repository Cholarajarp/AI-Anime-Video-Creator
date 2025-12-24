"""
Advanced Video Generator - FIXED VERSION
Uses imageio-ffmpeg for reliable video generation without infinite loops.
"""

import os
import uuid
import math
import random
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from loguru import logger


class AdvancedVideoGenerator:
    """Video generator that creates animated content."""

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)

    def generate_animated_video(
        self,
        prompt: str,
        audio_path: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        fps: int = 15,
        duration: float = 3.0,
        style: str = "anime",
        **kwargs
    ) -> Optional[str]:
        """Generate animated video with effects."""
        try:
            logger.info(f"🎨 Generating video: {duration:.1f}s @ {fps} FPS")

            num_frames = int(duration * fps)
            animation_type = self._analyze_prompt(prompt)
            logger.info(f"🎬 Animation type: {animation_type}")

            frames = self._generate_frames(prompt, num_frames, width, height, animation_type, style, fps)

            output_path = self._save_video(frames, audio_path, fps)

            if output_path:
                logger.info(f"✅ Video created: {output_path}")
                return str(output_path)

            return None

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _analyze_prompt(self, prompt: str) -> str:
        """Analyze prompt for animation style."""
        p = prompt.lower()

        if any(w in p for w in ['battle', 'fight', 'action', 'attack']):
            return 'action'
        elif any(w in p for w in ['magic', 'spell', 'fantasy', 'mystic']):
            return 'magic'
        elif any(w in p for w in ['cute', 'kawaii', 'chibi']):
            return 'cute'
        elif any(w in p for w in ['romantic', 'love', 'sunset']):
            return 'romantic'
        elif any(w in p for w in ['nature', 'forest', 'flower']):
            return 'nature'
        else:
            return 'anime'

    def _generate_frames(
        self,
        prompt: str,
        num_frames: int,
        width: int,
        height: int,
        animation_type: str,
        style: str,
        fps: int
    ) -> List[np.ndarray]:
        """Generate animated frames."""
        logger.info(f"🎨 Generating {num_frames} frames...")

        frames = []
        colors = self._get_colors(animation_type)

        for i in range(num_frames):
            if i % max(1, num_frames // 5) == 0:
                logger.info(f"   Frame {i+1}/{num_frames}")

            frame = self._create_frame(i, num_frames, width, height, colors, animation_type, prompt)
            frames.append(np.array(frame))

        return frames

    def _get_colors(self, style: str) -> dict:
        """Get color palette."""
        palettes = {
            'action': {'bg1': (40, 20, 60), 'bg2': (100, 40, 80), 'accent': (255, 100, 50), 'particle': (255, 150, 50)},
            'magic': {'bg1': (30, 20, 60), 'bg2': (80, 50, 120), 'accent': (200, 100, 255), 'particle': (180, 150, 255)},
            'cute': {'bg1': (255, 200, 220), 'bg2': (255, 180, 200), 'accent': (255, 100, 150), 'particle': (255, 200, 255)},
            'romantic': {'bg1': (120, 60, 80), 'bg2': (200, 100, 130), 'accent': (255, 150, 180), 'particle': (255, 180, 200)},
            'nature': {'bg1': (30, 80, 50), 'bg2': (60, 140, 80), 'accent': (100, 255, 150), 'particle': (150, 255, 150)},
            'anime': {'bg1': (50, 70, 130), 'bg2': (100, 120, 200), 'accent': (255, 150, 200), 'particle': (200, 180, 255)},
        }
        return palettes.get(style, palettes['anime'])

    def _create_frame(
        self,
        frame_idx: int,
        total_frames: int,
        width: int,
        height: int,
        colors: dict,
        style: str,
        prompt: str
    ) -> Image.Image:
        """Create a single animated frame."""
        t = frame_idx / max(total_frames - 1, 1)

        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img, 'RGBA')

        # Background
        self._draw_background(img, colors, t)

        # Character
        self._draw_character(draw, width, height, frame_idx, colors)

        # Particles
        self._draw_particles(draw, width, height, frame_idx, colors)

        # Prompt text
        if frame_idx < total_frames * 0.7:
            self._draw_text(draw, prompt, width, height)

        return img

    def _draw_background(self, img: Image.Image, colors: dict, t: float):
        """Draw gradient background."""
        width, height = img.size
        bg1 = colors['bg1']
        bg2 = colors['bg2']

        # Vectorized gradient
        y_coords = np.arange(height).reshape(-1, 1)
        offset = int(t * 50) % 50
        wave = np.sin((y_coords + offset) * 0.05) * 0.1

        gradient = (y_coords + wave * height) / height
        gradient = np.clip(gradient, 0, 1)

        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(3):
            pixels[:, :, i] = (bg1[i] * (1 - gradient) + bg2[i] * gradient).astype(np.uint8)

        img.paste(Image.fromarray(pixels), (0, 0))

    def _draw_character(self, draw, width: int, height: int, frame_idx: int, colors: dict):
        """Draw animated character."""
        center_x = width // 2
        center_y = int(height * 0.55)

        # Breathing animation
        t = frame_idx / 24  # Assuming ~24 fps
        breathe = math.sin(t * math.pi * 2) * 10
        sway = math.sin(t * math.pi * 1.5) * 8

        # Body
        body_w = int(width * 0.18)
        body_h = int(height * 0.26)
        draw.ellipse(
            [center_x - body_w + int(sway * 0.5), center_y - body_h + int(breathe),
             center_x + body_w + int(sway * 0.5), center_y + body_h + int(breathe)],
            fill=colors['accent']
        )

        # Head
        head_r = int(width * 0.1)
        head_y = center_y - body_h - head_r + int(breathe * 0.3)
        draw.ellipse(
            [center_x - head_r + int(sway), head_y - head_r,
             center_x + head_r + int(sway), head_y + head_r],
            fill=colors['accent']
        )

        # Eyes with blink
        blink = abs(math.sin(t * math.pi * 0.5))
        eye_h = max(2, int(8 * blink))
        eye_spacing = head_r // 2

        for side in [-1, 1]:
            eye_x = center_x + side * eye_spacing + int(sway)
            draw.ellipse(
                [eye_x - 6, head_y - eye_h,
                 eye_x + 6, head_y + eye_h],
                fill=(255, 255, 255)
            )
            if blink > 0.3:
                draw.ellipse(
                    [eye_x - 3, head_y - eye_h * 0.5,
                     eye_x + 3, head_y + eye_h * 0.5],
                    fill=(40, 40, 60)
                )

        # Hair
        hair_sway = math.sin(t * math.pi * 2.5) * 6
        hair_color = tuple(max(0, c - 40) for c in colors['accent'])
        draw.polygon([
            (center_x - head_r + int(sway + hair_sway), head_y - head_r),
            (center_x + int(sway + hair_sway * 0.5), head_y - head_r * 1.7),
            (center_x + head_r + int(sway + hair_sway), head_y - head_r),
        ], fill=hair_color)

    def _draw_particles(self, draw, width: int, height: int, frame_idx: int, colors: dict):
        """Draw floating particles."""
        random.seed(42)
        for i in range(20):
            speed = random.random() * 2 + 0.5
            start_x = random.random() * width
            start_y = random.random() * height

            x = (start_x + frame_idx * random.random() * 2) % width
            y = (start_y - frame_idx * speed * 2) % height

            size = 2 + random.random() * 3
            alpha = int(100 + math.sin(frame_idx * 0.1 + i) * 50)

            draw.ellipse(
                [x - size * 1.5, y - size * 1.5, x + size * 1.5, y + size * 1.5],
                fill=colors['particle'] + (alpha // 3,)
            )
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=colors['particle'] + (alpha,)
            )

    def _draw_text(self, draw, prompt: str, width: int, height: int):
        """Draw prompt text."""
        text = prompt[:50] + "..." if len(prompt) > 50 else prompt
        y = height - 40
        padding = 8

        # Simple text box
        draw.rectangle([10, y - padding, width - 10, y + 20 + padding], fill=(0, 0, 0, 180))
        draw.text((15, y), text, fill=(255, 255, 255))

    def _save_video(self, frames: List[np.ndarray], audio_path: Optional[str], fps: int) -> Optional[str]:
        """Save video using imageio-ffmpeg ONLY (no moviepy)."""
        try:
            import imageio

            output_path = self.output_dir / f"video_{uuid.uuid4().hex[:8]}.mp4"
            temp_video = self.temp_dir / f"temp_{uuid.uuid4().hex[:8]}.mp4"

            logger.info(f"💾 Saving video: {len(frames)} frames...")

            # Save frames to video
            writer = imageio.get_writer(
                str(temp_video),
                fps=fps,
                codec='libx264',
                quality=8,
                pixelformat='yuv420p',
                macro_block_size=1
            )

            for frame in frames:
                writer.append_data(frame)
            writer.close()

            logger.info("✅ Video frames saved")

            # Merge audio using ffmpeg (NO moviepy, NO retries)
            if audio_path and os.path.exists(audio_path):
                try:
                    from imageio_ffmpeg import get_ffmpeg_exe
                    import subprocess

                    ffmpeg = get_ffmpeg_exe()
                    logger.info("🎵 Merging audio...")

                    cmd = [
                        ffmpeg, '-y',
                        '-i', str(temp_video),
                        '-i', audio_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-shortest',
                        str(output_path)
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                    if result.returncode == 0 and output_path.exists():
                        logger.info("✅ Audio merged successfully")
                        temp_video.unlink(missing_ok=True)
                        return str(output_path)
                    else:
                        logger.warning(f"Audio merge warning: {result.stderr[:100] if result.stderr else 'unknown'}")

                except Exception as e:
                    logger.warning(f"Audio merge failed: {e}")

            # Fallback: use video without audio
            import shutil
            if temp_video.exists():
                shutil.move(str(temp_video), str(output_path))
            logger.info("✅ Video saved (without audio)")

            return str(output_path) if output_path.exists() else None

        except Exception as e:
            logger.error(f"Save error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

