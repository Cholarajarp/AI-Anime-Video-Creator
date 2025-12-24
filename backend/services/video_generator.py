"""
Video Generation Service
Handles video creation from images with advanced features
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, List
import uuid
from loguru import logger


class VideoGenerator:
    """Generates videos from prompts using various methods."""

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir = Path("./temp")
        self.temp_dir.mkdir(exist_ok=True)

    def generate_placeholder_video(
        self,
        prompt: str,
        audio_path: str,
        width: int = 512,
        height: int = 512,
        fps: int = 15,
        duration: float = 3.0,
        **kwargs
    ) -> Optional[str]:
        """
        Generate a placeholder video with audio.
        This creates a simple animated video when ComfyUI is not available.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
            import numpy as np

            # Calculate frames
            num_frames = int(duration * fps)

            logger.info(f"Generating placeholder video: {num_frames} frames at {fps} FPS")

            # Generate frames
            frames = []
            colors = self._get_anime_color_palette(prompt)

            for i in range(num_frames):
                # Create gradient background
                img = self._create_gradient_frame(width, height, i, num_frames, colors)

                # Add animated elements
                img = self._add_animated_elements(img, i, num_frames, prompt)

                frames.append(np.array(img))

            # Save as video
            output_path = self.output_dir / f"video_{uuid.uuid4().hex[:8]}.mp4"

            # Try using imageio (works without ffmpeg executable)
            try:
                import imageio

                # Save frames as video
                temp_video = self.temp_dir / f"temp_{uuid.uuid4().hex[:8]}.mp4"
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

                # Merge with audio using ffmpeg-python
                if audio_path and os.path.exists(audio_path):
                    self._merge_audio_video(str(temp_video), audio_path, str(output_path))
                else:
                    # No audio, just rename
                    import shutil
                    shutil.move(str(temp_video), str(output_path))

                logger.info(f"Video saved to: {output_path}")
                return str(output_path)

            except Exception as e:
                logger.error(f"Video generation failed: {e}")
                return None

        except Exception as e:
            logger.error(f"Placeholder video generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _create_gradient_frame(
        self,
        width: int,
        height: int,
        frame_idx: int,
        total_frames: int,
        colors: List[tuple]
    ) -> Image.Image:
        """Create animated gradient background."""
        from PIL import Image
        import math

        img = Image.new('RGB', (width, height))
        pixels = img.load()

        # Animated offset
        offset = (frame_idx / total_frames) * math.pi * 2

        for y in range(height):
            for x in range(width):
                # Create animated gradient
                t = (x / width + y / height) / 2
                t = (t + offset / (math.pi * 2)) % 1.0

                # Interpolate between colors
                color_idx = int(t * (len(colors) - 1))
                next_idx = min(color_idx + 1, len(colors) - 1)
                local_t = (t * (len(colors) - 1)) % 1.0

                c1 = colors[color_idx]
                c2 = colors[next_idx]

                r = int(c1[0] * (1 - local_t) + c2[0] * local_t)
                g = int(c1[1] * (1 - local_t) + c2[1] * local_t)
                b = int(c1[2] * (1 - local_t) + c2[2] * local_t)

                pixels[x, y] = (r, g, b)

        return img

    def _add_animated_elements(
        self,
        img: Image.Image,
        frame_idx: int,
        total_frames: int,
        prompt: str
    ) -> Image.Image:
        """Add animated elements to the frame."""
        from PIL import ImageDraw, ImageFont
        import math

        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Pulsing circle
        t = frame_idx / total_frames
        radius = int(50 + 20 * math.sin(t * math.pi * 4))
        center_x = int(width / 2 + 100 * math.cos(t * math.pi * 2))
        center_y = int(height / 2 + 100 * math.sin(t * math.pi * 2))

        # Draw glow effect
        for i in range(3, 0, -1):
            alpha_val = int(100 / i)
            glow_radius = radius + i * 10
            draw.ellipse(
                [
                    center_x - glow_radius,
                    center_y - glow_radius,
                    center_x + glow_radius,
                    center_y + glow_radius
                ],
                fill=(255, 255, 255, alpha_val)
            )

        draw.ellipse(
            [
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius
            ],
            fill=(255, 220, 255)
        )

        # Add text overlay (truncated prompt)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        text = prompt[:50] + "..." if len(prompt) > 50 else prompt

        # Text with shadow
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height - 80

        # Shadow
        draw.text((text_x + 2, text_y + 2), text, fill=(0, 0, 0), font=font)
        # Main text
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

        return img

    def _get_anime_color_palette(self, prompt: str) -> List[tuple]:
        """Get color palette based on prompt keywords."""
        palettes = {
            'default': [(100, 100, 255), (255, 100, 200), (255, 200, 100)],
            'fire': [(255, 100, 0), (255, 200, 0), (255, 50, 50)],
            'water': [(0, 100, 255), (100, 200, 255), (200, 220, 255)],
            'nature': [(50, 200, 50), (100, 255, 100), (200, 255, 150)],
            'dark': [(50, 0, 100), (100, 0, 150), (80, 0, 120)],
            'light': [(255, 250, 200), (255, 200, 250), (200, 250, 255)],
            'fantasy': [(200, 100, 255), (255, 150, 200), (150, 200, 255)],
        }

        prompt_lower = prompt.lower()

        for keyword, palette in palettes.items():
            if keyword in prompt_lower:
                return palette

        return palettes['default']

    def _merge_audio_video(self, video_path: str, audio_path: str, output_path: str):
        """Merge audio and video using ffmpeg-python."""
        try:
            import ffmpeg

            # Input streams
            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(audio_path)

            # Merge
            (
                ffmpeg
                .output(
                    video,
                    audio,
                    output_path,
                    vcodec='copy',
                    acodec='aac',
                    strict='experimental',
                    shortest=None
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            logger.info(f"Audio merged successfully: {output_path}")

        except Exception as e:
            logger.error(f"Audio merge failed: {e}, using video without audio")
            # Copy video without audio
            import shutil
            shutil.copy(video_path, output_path)

    def generate_with_comfyui(
        self,
        prompt: str,
        negative_prompt: str,
        audio_path: str,
        width: int,
        height: int,
        fps: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        checkpoint: str,
        motion_module: str,
        frame_count: int
    ) -> Optional[str]:
        """Generate video using ComfyUI API."""
        try:
            from backend.services.comfyui_client import ComfyUIClient

            client = ComfyUIClient()

            if not client.is_connected():
                logger.warning("ComfyUI not available, using placeholder")
                return None

            # Load workflow template
            workflow_path = Path("./workflows/animatediff_base.json")
            if not workflow_path.exists():
                logger.error("Workflow template not found")
                return None

            import json
            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            # Modify workflow parameters
            # This is a simplified example - actual implementation depends on workflow structure
            workflow = self._update_workflow_params(
                workflow,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                checkpoint=checkpoint,
                motion_module=motion_module,
                frame_count=frame_count
            )

            # Queue the workflow
            result = client.queue_prompt(workflow)

            if result and 'output_path' in result:
                video_path = result['output_path']

                # Merge with audio
                final_output = self.output_dir / f"video_{uuid.uuid4().hex[:8]}.mp4"
                self._merge_audio_video(video_path, audio_path, str(final_output))

                return str(final_output)

            return None

        except Exception as e:
            logger.error(f"ComfyUI generation failed: {e}")
            return None

    def _update_workflow_params(self, workflow: dict, **params) -> dict:
        """Update workflow parameters."""
        # This is a placeholder - actual implementation depends on workflow structure
        # You would need to traverse the workflow JSON and update the appropriate nodes
        return workflow

