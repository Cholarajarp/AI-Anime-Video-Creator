"""
Simple Working Video Generator
Creates animated videos with visible anime-style content
"""

import os
import uuid
import math
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from loguru import logger


class SimpleVideoGenerator:
    """Simple video generator that definitely works."""

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)

    def generate_video(
        self,
        prompt: str,
        audio_path: str,
        width: int = 512,
        height: int = 512,
        fps: int = 15,
        duration: float = 3.0
    ) -> Optional[str]:
        """Generate animated video."""
        try:
            num_frames = max(int(duration * fps), 30)  # At least 30 frames
            logger.info(f"🎬 Generating {num_frames} frames @ {fps} FPS")

            # Detect animation style from prompt
            style = self._detect_style(prompt)
            logger.info(f"🎨 Style detected: {style}")

            # Generate all frames
            frames = []
            for i in range(num_frames):
                frame = self._create_frame(prompt, i, num_frames, width, height, style)
                frames.append(np.array(frame))

                if i % 20 == 0:
                    logger.info(f"   Frame {i+1}/{num_frames}")

            logger.info(f"✅ All {num_frames} frames generated")

            # Save video
            output_path = self._save_video(frames, audio_path, fps)

            return output_path

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _detect_style(self, prompt: str) -> str:
        """Detect animation style from prompt."""
        p = prompt.lower()

        if any(w in p for w in ['battle', 'fight', 'action', 'attack', 'power']):
            return 'action'
        elif any(w in p for w in ['magic', 'spell', 'glow', 'mystic', 'fantasy']):
            return 'magic'
        elif any(w in p for w in ['cute', 'kawaii', 'chibi', 'happy', 'smile']):
            return 'cute'
        elif any(w in p for w in ['dark', 'night', 'shadow', 'mystery']):
            return 'dark'
        elif any(w in p for w in ['nature', 'forest', 'flower', 'garden']):
            return 'nature'
        elif any(w in p for w in ['sunset', 'romantic', 'love', 'gentle']):
            return 'romantic'
        else:
            return 'anime'

    def _create_frame(
        self,
        prompt: str,
        frame_idx: int,
        total_frames: int,
        width: int,
        height: int,
        style: str
    ) -> Image.Image:
        """Create a single animated frame."""

        # Progress through animation (0 to 1) - USE FRAME INDEX FOR ANIMATION
        t = frame_idx / max(total_frames - 1, 1)

        # Create base image
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img, 'RGBA')

        # Get colors based on style
        colors = self._get_style_colors(style)

        # Draw animated background (uses frame_idx internally now)
        self._draw_background(img, t, style, colors)

        # Draw main subject (anime character)
        self._draw_character(draw, width, height, t, style, colors)

        # Draw effects (uses frame_idx for animation)
        self._draw_effects(draw, width, height, frame_idx, total_frames, style, colors)

        # Draw particles (frame-based animation)
        self._draw_particles(draw, width, height, frame_idx, total_frames, style, colors)

        # Add prompt text at bottom (fade in at start)
        if frame_idx < total_frames * 0.7:  # Show for 70% of video
            self._draw_text(draw, prompt, width, height, t)

        return img

    def _get_style_colors(self, style: str) -> dict:
        """Get color palette for style."""
        palettes = {
            'action': {
                'bg1': (30, 20, 50),
                'bg2': (80, 40, 100),
                'accent': (255, 100, 50),
                'glow': (255, 200, 100),
                'particle': (255, 150, 50),
            },
            'magic': {
                'bg1': (20, 10, 60),
                'bg2': (80, 40, 150),
                'accent': (200, 100, 255),
                'glow': (255, 200, 255),
                'particle': (180, 150, 255),
            },
            'cute': {
                'bg1': (255, 200, 220),
                'bg2': (255, 150, 200),
                'accent': (255, 100, 150),
                'glow': (255, 255, 200),
                'particle': (255, 200, 255),
            },
            'dark': {
                'bg1': (10, 10, 20),
                'bg2': (30, 30, 60),
                'accent': (100, 100, 200),
                'glow': (150, 150, 255),
                'particle': (100, 150, 200),
            },
            'nature': {
                'bg1': (20, 80, 40),
                'bg2': (40, 150, 80),
                'accent': (100, 255, 150),
                'glow': (200, 255, 200),
                'particle': (150, 255, 150),
            },
            'romantic': {
                'bg1': (100, 50, 80),
                'bg2': (200, 100, 150),
                'accent': (255, 150, 180),
                'glow': (255, 200, 220),
                'particle': (255, 180, 200),
            },
            'anime': {
                'bg1': (40, 60, 120),
                'bg2': (100, 120, 200),
                'accent': (255, 150, 200),
                'glow': (255, 220, 255),
                'particle': (200, 180, 255),
            },
        }
        return palettes.get(style, palettes['anime'])

    def _draw_background(self, img: Image.Image, t: float, style: str, colors: dict):
        """Draw animated gradient background - FAST VERSION."""
        width, height = img.size

        bg1 = colors['bg1']
        bg2 = colors['bg2']

        # Create gradient using numpy for speed
        import numpy as np

        # Animated offset
        offset = int(t * 50) % 50

        # Create gradient array
        gradient = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            # Animated wave
            wave = math.sin((y + offset) * 0.05) * 20
            blend = (y + wave) / height
            blend = max(0, min(1, blend))

            r = int(bg1[0] * (1 - blend) + bg2[0] * blend)
            g = int(bg1[1] * (1 - blend) + bg2[1] * blend)
            b = int(bg1[2] * (1 - blend) + bg2[2] * blend)

            gradient[y, :] = [r, g, b]

        # Apply to image
        img.paste(Image.fromarray(gradient), (0, 0))

    def _draw_character(self, draw, width: int, height: int, t: float, style: str, colors: dict):
        """Draw anime character silhouette with animation."""

        center_x = width // 2
        center_y = int(height * 0.55)

        # Breathing animation
        breathe = math.sin(t * math.pi * 4) * 5

        # Body (oval)
        body_width = int(width * 0.25)
        body_height = int(height * 0.35)
        draw.ellipse(
            [center_x - body_width, center_y - body_height + breathe,
             center_x + body_width, center_y + body_height + breathe],
            fill=colors['accent']
        )

        # Head (circle)
        head_size = int(width * 0.15)
        head_y = center_y - body_height - head_size + breathe
        draw.ellipse(
            [center_x - head_size, head_y - head_size,
             center_x + head_size, head_y + head_size],
            fill=colors['accent']
        )

        # Eyes (white with colored pupils)
        eye_size = int(head_size * 0.4)
        eye_y = head_y - eye_size // 2

        # Left eye
        draw.ellipse(
            [center_x - head_size//2 - eye_size//2, eye_y - eye_size//2,
             center_x - head_size//2 + eye_size//2, eye_y + eye_size//2],
            fill=(255, 255, 255)
        )
        # Right eye
        draw.ellipse(
            [center_x + head_size//2 - eye_size//2, eye_y - eye_size//2,
             center_x + head_size//2 + eye_size//2, eye_y + eye_size//2],
            fill=(255, 255, 255)
        )

        # Pupils (animated - looking around)
        pupil_size = eye_size // 3
        look_x = math.sin(t * math.pi * 2) * (eye_size // 4)
        look_y = math.cos(t * math.pi * 3) * (eye_size // 6)

        # Left pupil
        draw.ellipse(
            [center_x - head_size//2 - pupil_size//2 + look_x,
             eye_y - pupil_size//2 + look_y,
             center_x - head_size//2 + pupil_size//2 + look_x,
             eye_y + pupil_size//2 + look_y],
            fill=colors['bg1']
        )
        # Right pupil
        draw.ellipse(
            [center_x + head_size//2 - pupil_size//2 + look_x,
             eye_y - pupil_size//2 + look_y,
             center_x + head_size//2 + pupil_size//2 + look_x,
             eye_y + pupil_size//2 + look_y],
            fill=colors['bg1']
        )

        # Hair (triangles on top)
        hair_color = self._adjust_color(colors['accent'], 0.7)
        points = [
            (center_x - head_size, head_y - head_size),
            (center_x, head_y - head_size * 1.8),
            (center_x + head_size, head_y - head_size),
        ]
        draw.polygon(points, fill=hair_color)

        # Side hair
        draw.polygon([
            (center_x - head_size, head_y - head_size//2),
            (center_x - head_size - head_size//2, head_y + head_size//2),
            (center_x - head_size, head_y + head_size//2),
        ], fill=hair_color)

        draw.polygon([
            (center_x + head_size, head_y - head_size//2),
            (center_x + head_size + head_size//2, head_y + head_size//2),
            (center_x + head_size, head_y + head_size//2),
        ], fill=hair_color)

    def _draw_effects(self, draw, width: int, height: int, frame_idx: int, total_frames: int, style: str, colors: dict):
        """Draw style-specific effects with FRAME-BASED animation."""

        t = frame_idx / max(total_frames - 1, 1)

        if style == 'magic':
            # Magic circle - rotating
            center_x, center_y = width // 2, int(height * 0.65)
            radius = int(min(width, height) * 0.3)

            # Draw rotating magic circle
            num_points = 8
            for i in range(num_points):
                # Rotate based on frame
                angle = (i / num_points) * math.pi * 2 + (frame_idx * 0.1)
                x = center_x + math.cos(angle) * radius
                y = center_y + math.sin(angle) * radius

                # Pulsing size
                size = 8 + math.sin(frame_idx * 0.2 + i) * 4

                draw.ellipse(
                    [x - size, y - size, x + size, y + size],
                    fill=colors['glow'] + (180,)
                )

            # Inner circle
            inner_radius = radius * 0.6
            draw.ellipse(
                [center_x - inner_radius, center_y - inner_radius,
                 center_x + inner_radius, center_y + inner_radius],
                outline=colors['glow'] + (100,),
                width=2
            )

        elif style == 'action':
            # Speed lines - animate from right to left
            for i in range(12):
                offset = (frame_idx * 15 + i * 40) % (width + 200) - 100
                start_x = width - offset
                start_y = height // 4 + i * (height // 15)
                length = 80 + (i % 3) * 30

                if 0 < start_x < width:
                    draw.line(
                        [start_x, start_y, start_x - length, start_y + length * 0.1],
                        fill=colors['glow'] + (150,),
                        width=3
                    )

        elif style == 'romantic' or style == 'cute':
            # Floating hearts
            for i in range(6):
                # Each heart moves independently
                x = (i * 80 + frame_idx * (2 + i * 0.3)) % width
                y = height - ((frame_idx * (3 + i * 0.5) + i * 60) % (height * 0.7))

                size = 12 + math.sin(frame_idx * 0.15 + i) * 4
                alpha = int(180 + math.sin(frame_idx * 0.1 + i) * 50)

                # Draw heart shape
                self._draw_heart(draw, x, y, size, colors['glow'] + (alpha,))

        elif style == 'nature':
            # Falling leaves/petals
            for i in range(10):
                x = (i * 50 + frame_idx * (1 + i * 0.2) + math.sin(frame_idx * 0.1 + i) * 20) % width
                y = (frame_idx * (2 + i * 0.3) + i * 40) % height

                size = 6 + math.sin(frame_idx * 0.2 + i) * 2
                rotation = frame_idx * 0.1 + i

                # Simple petal
                draw.ellipse(
                    [x - size, y - size//2, x + size, y + size//2],
                    fill=colors['particle'] + (150,)
                )

    def _draw_heart(self, draw, x: float, y: float, size: float, color: tuple):
        """Draw a simple heart shape."""
        # Heart using two circles and a triangle
        r = size / 2
        # Left circle
        draw.ellipse([x - r, y - r, x, y + r//2], fill=color)
        # Right circle
        draw.ellipse([x, y - r, x + r, y + r//2], fill=color)
        # Bottom triangle
        draw.polygon([
            (x - r, y),
            (x + r, y),
            (x, y + r * 1.5)
        ], fill=color)

    def _draw_particles(self, draw, width: int, height: int, frame_idx: int, total_frames: int, style: str, colors: dict):
        """Draw floating particles with VISIBLE movement."""
        import random

        num_particles = 25
        t = frame_idx / max(total_frames - 1, 1)

        for i in range(num_particles):
            # Each particle has unique but consistent behavior
            random.seed(42 + i)
            speed_x = random.random() * 2 - 1  # -1 to 1
            speed_y = random.random() * 1.5 + 0.5  # 0.5 to 2 (upward)
            start_x = random.random() * width
            start_y = random.random() * height

            # Calculate position based on frame
            x = (start_x + frame_idx * speed_x * 3) % width
            y = (start_y - frame_idx * speed_y * 4) % height  # Move up

            # Size pulsing
            base_size = random.random() * 4 + 2
            size = base_size + math.sin(frame_idx * 0.3 + i) * 2

            # Alpha pulsing
            alpha = int(150 + math.sin(frame_idx * 0.2 + i * 0.5) * 80)

            # Draw with glow
            glow_size = size + 3
            draw.ellipse(
                [x - glow_size, y - glow_size, x + glow_size, y + glow_size],
                fill=colors['particle'] + (alpha // 3,)
            )
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=colors['particle'] + (alpha,)
            )

    def _draw_text(self, draw, prompt: str, width: int, height: int, t: float):
        """Draw prompt text at bottom."""
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Truncate prompt
        text = prompt[:60] + "..." if len(prompt) > 60 else prompt

        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (width - text_width) // 2
        y = height - text_height - 30

        # Background
        padding = 10
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(0, 0, 0, 180)
        )

        # Text
        draw.text((x + 1, y + 1), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    def _adjust_color(self, color: tuple, factor: float) -> tuple:
        """Darken or lighten a color."""
        return tuple(max(0, min(255, int(c * factor))) for c in color)

    def _save_video(self, frames: List[np.ndarray], audio_path: str, fps: int) -> Optional[str]:
        """Save frames as video with audio."""
        try:
            import imageio

            output_filename = f"video_{uuid.uuid4().hex[:8]}.mp4"
            output_path = self.output_dir / output_filename
            temp_video = self.temp_dir / f"temp_{uuid.uuid4().hex[:8]}.mp4"

            logger.info(f"💾 Saving video to {output_path}")

            # Save frames as video
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

            # Merge audio using ffmpeg
            if audio_path and os.path.exists(audio_path):
                try:
                    from imageio_ffmpeg import get_ffmpeg_exe
                    import subprocess

                    ffmpeg = get_ffmpeg_exe()
                    logger.info(f"🎵 Merging audio...")

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
                        logger.info("✅ Audio merged!")
                        temp_video.unlink(missing_ok=True)
                        return str(output_path)
                    else:
                        logger.warning(f"FFmpeg warning: {result.stderr[:200] if result.stderr else 'unknown'}")

                except Exception as e:
                    logger.warning(f"Audio merge failed: {e}")

            # Fallback: use video without audio
            import shutil
            shutil.move(str(temp_video), str(output_path))
            logger.info("✅ Video saved (without audio)")

            return str(output_path)

        except Exception as e:
            logger.error(f"Save error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


# More templates
ANIME_TEMPLATES = {
    "Anime Girl - Cheerful 🎀": "1girl, anime, cheerful, smiling, pink hair, school uniform, sparkles, happy, cute",
    "Anime Boy - Cool 😎": "1boy, anime, cool, confident, dark hair, jacket, urban background, stylish",
    "Magical Girl ✨": "magical girl, anime, transformation, glowing, sparkles, magical staff, fantasy, beautiful",
    "Battle Warrior ⚔️": "anime warrior, battle stance, dynamic, sword, action, intense, powerful, dramatic lighting",
    "Cute Chibi 🧸": "chibi, kawaii, big eyes, cute, happy, pastel colors, adorable, simple background",
    "Dark Fantasy 🌙": "anime, dark fantasy, mysterious, night, moon, gothic, elegant, shadows",
    "Nature Spirit 🌸": "anime, nature spirit, forest, flowers, peaceful, ethereal, magical, beautiful scenery",
    "Romantic Sunset 💕": "anime, romantic, sunset, gentle, warm colors, emotional, beautiful, couple silhouette",
    "Cyberpunk Neon 🌃": "anime, cyberpunk, neon lights, futuristic, night city, rain, purple blue tones",
    "Traditional Japanese 🏯": "anime, traditional, kimono, japanese, elegant, cherry blossoms, graceful",
    "Action Hero 💥": "anime hero, dynamic pose, energy aura, powerful, action effects, dramatic",
    "Peaceful Garden 🌺": "anime, peaceful garden, flowers, sunshine, relaxing, nature, beautiful, gentle",
    "Space Adventure 🚀": "anime, space, stars, galaxy, astronaut, sci-fi, cosmic, adventure",
    "Winter Wonderland ❄️": "anime, winter, snow, cold, cozy, warm clothes, peaceful, beautiful scenery",
    "Summer Beach 🏖️": "anime, summer, beach, ocean, sunny, happy, vacation, fun",
}

