"""
AnimeTemplateComposer - ComfyUI Custom Nodes
Composite template layers with animation
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
import json


class AnimeTemplateComposer:
    """
    ComfyUI custom node for compositing anime template layers
    with per-frame animation transforms
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background_latent": ("LATENT",),
                "character_latent": ("LATENT",),
                "frame_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999
                }),
            },
            "optional": {
                "mouth_open_latent": ("LATENT",),
                "mouth_closed_latent": ("LATENT",),
                "eyes_open_latent": ("LATENT",),
                "eyes_closed_latent": ("LATENT",),
                "animation_map": ("STRING", {
                    "default": "{}",
                    "multiline": True
                }),
                "blend_mode": (["normal", "overlay", "multiply", "screen"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("composited_latent",)
    FUNCTION = "compose_frame"
    CATEGORY = "animation/template"

    def compose_frame(
        self,
        background_latent: Dict[str, torch.Tensor],
        character_latent: Dict[str, torch.Tensor],
        frame_index: int,
        mouth_open_latent: Optional[Dict[str, torch.Tensor]] = None,
        mouth_closed_latent: Optional[Dict[str, torch.Tensor]] = None,
        eyes_open_latent: Optional[Dict[str, torch.Tensor]] = None,
        eyes_closed_latent: Optional[Dict[str, torch.Tensor]] = None,
        animation_map: str = "{}",
        blend_mode: str = "normal"
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Composite layers for a single frame"""

        # Parse animation map
        try:
            anim_data = json.loads(animation_map)
            frame_transform = anim_data.get(str(frame_index), {})
        except json.JSONDecodeError:
            frame_transform = {}

        # Get base tensors
        bg = background_latent["samples"].clone()
        char = character_latent["samples"].clone()

        # Apply transforms to character
        char = self._apply_transform(char, frame_transform)

        # Determine which mouth/eye variant to use
        mouth_state = frame_transform.get("mouth", "closed")
        eye_state = frame_transform.get("eye_state", "open")

        # Select mouth latent
        if mouth_state == "open" and mouth_open_latent is not None:
            mouth = mouth_open_latent["samples"]
        elif mouth_closed_latent is not None:
            mouth = mouth_closed_latent["samples"]
        else:
            mouth = None

        # Select eye latent
        if eye_state in ["closed", "closing"] and eyes_closed_latent is not None:
            eyes = eyes_closed_latent["samples"]
        elif eyes_open_latent is not None:
            eyes = eyes_open_latent["samples"]
        else:
            eyes = None

        # Composite layers in Z-order
        result = bg.clone()

        # Add character layer
        result = self._blend_layers(result, char, blend_mode, 1.0)

        # Add eyes layer
        if eyes is not None:
            eyes = self._apply_transform(eyes, frame_transform)
            result = self._blend_layers(result, eyes, blend_mode, 1.0)

        # Add mouth layer
        if mouth is not None:
            mouth = self._apply_transform(mouth, frame_transform)
            result = self._blend_layers(result, mouth, blend_mode, 1.0)

        return ({"samples": result},)

    def _apply_transform(
        self,
        latent: torch.Tensor,
        transform: Dict[str, Any]
    ) -> torch.Tensor:
        """Apply spatial transforms to latent"""

        scale = transform.get("scale", 1.0)
        offset_x = transform.get("offset_x", 0.0)
        offset_y = transform.get("offset_y", 0.0)
        rotation = transform.get("rotation", 0.0)

        if scale == 1.0 and offset_x == 0.0 and offset_y == 0.0 and rotation == 0.0:
            return latent

        # Build affine transform matrix
        batch_size = latent.shape[0]

        # Scale and rotation
        cos_r = np.cos(np.radians(rotation))
        sin_r = np.sin(np.radians(rotation))

        theta = torch.tensor([
            [scale * cos_r, -scale * sin_r, offset_x / latent.shape[3]],
            [scale * sin_r, scale * cos_r, offset_y / latent.shape[2]]
        ], dtype=latent.dtype, device=latent.device)

        theta = theta.unsqueeze(0).repeat(batch_size, 1, 1)

        # Apply transform
        grid = torch.nn.functional.affine_grid(
            theta,
            latent.size(),
            align_corners=False
        )

        transformed = torch.nn.functional.grid_sample(
            latent,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )

        return transformed

    def _blend_layers(
        self,
        base: torch.Tensor,
        overlay: torch.Tensor,
        mode: str,
        opacity: float
    ) -> torch.Tensor:
        """Blend two latent layers"""

        if mode == "normal":
            return base * (1 - opacity) + overlay * opacity

        elif mode == "overlay":
            mask = base < 0
            result = torch.where(
                mask,
                2 * base * overlay,
                1 - 2 * (1 - base) * (1 - overlay)
            )
            return base * (1 - opacity) + result * opacity

        elif mode == "multiply":
            result = base * overlay
            return base * (1 - opacity) + result * opacity

        elif mode == "screen":
            result = 1 - (1 - base) * (1 - overlay)
            return base * (1 - opacity) + result * opacity

        return base


class AnimeTemplateLoader:
    """Load template pack for animation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "animation_mode": (["idle", "talking", "emotional"],),
                "target_width": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "target_height": ("INT", {"default": 512, "min": 64, "max": 2048}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "LATENT", "LATENT")
    RETURN_NAMES = (
        "background", "character",
        "mouth_open", "mouth_closed",
        "eyes_open", "eyes_closed"
    )
    FUNCTION = "load_template"
    CATEGORY = "animation/template"

    def load_template(
        self,
        template_path: str,
        animation_mode: str = "idle",
        target_width: int = 512,
        target_height: int = 512
    ):
        """Load all template layers"""
        try:
            from backend.services.template_loader import TemplateLoader

            loader = TemplateLoader()
            templates = loader.load_template_pack(
                template_path,
                target_resolution=(target_width, target_height),
                animation_mode=animation_mode
            )

            # Convert to latent format
            def to_latent(key):
                if key in templates and isinstance(templates[key], torch.Tensor):
                    # Add batch dimension and convert to latent-like format
                    tensor = templates[key]
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    return {"samples": tensor}
                return {"samples": torch.zeros(1, 4, target_height // 8, target_width // 8)}

            return (
                to_latent("background"),
                to_latent("character"),
                to_latent("mouth_open"),
                to_latent("mouth_closed"),
                to_latent("eyes_open"),
                to_latent("eyes_closed"),
            )

        except Exception as e:
            # Return empty latents on error
            empty = {"samples": torch.zeros(1, 4, target_height // 8, target_width // 8)}
            return (empty, empty, empty, empty, empty, empty)


class AnimationMapGenerator:
    """Generate animation maps for template animation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frame_count": ("INT", {"default": 32, "min": 1, "max": 9999}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            },
            "optional": {
                "audio_path": ("STRING", {"default": ""}),
                "include_idle": ("BOOLEAN", {"default": True}),
                "include_mouth_sync": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("animation_map",)
    FUNCTION = "generate_map"
    CATEGORY = "animation/template"

    def generate_map(
        self,
        frame_count: int,
        fps: int,
        audio_path: str = "",
        include_idle: bool = True,
        include_mouth_sync: bool = True,
        seed: int = 42
    ) -> Tuple[str]:
        """Generate combined animation map"""

        animation_map = {}

        # Generate idle animation
        if include_idle:
            try:
                from backend.services.idle_animation import generate_idle_animation
                idle_map = generate_idle_animation(frame_count, fps, seed)
                for frame, data in idle_map.items():
                    animation_map[frame] = data
            except ImportError:
                pass
            except Exception as e:
                print(f"Error generating idle animation: {e}")

        # Generate mouth sync
        if include_mouth_sync and audio_path:
            try:
                from backend.services.mouth_sync import generate_mouth_animation
                mouth_map, _ = generate_mouth_animation(audio_path, fps)
                for mouth_data in mouth_map:
                    frame = mouth_data["frame"]
                    if frame in animation_map:
                        animation_map[frame]["mouth"] = mouth_data["mouth"]
                        animation_map[frame]["mouth_intensity"] = mouth_data["intensity"]
                    else:
                        animation_map[frame] = mouth_data
            except ImportError:
                pass
            except Exception as e:
                print(f"Error generating mouth sync: {e}")

        return (json.dumps(animation_map),)


class TemplateFrameBatcher:
    """Batch process frames with template animation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("LATENT",),
                "character": ("LATENT",),
                "animation_map": ("STRING",),
                "frame_count": ("INT", {"default": 32, "min": 1, "max": 9999}),
            },
            "optional": {
                "mouth_open": ("LATENT",),
                "mouth_closed": ("LATENT",),
                "eyes_open": ("LATENT",),
                "eyes_closed": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("batched_latents",)
    FUNCTION = "batch_frames"
    CATEGORY = "animation/template"

    def batch_frames(
        self,
        background: Dict[str, torch.Tensor],
        character: Dict[str, torch.Tensor],
        animation_map: str,
        frame_count: int,
        mouth_open: Optional[Dict[str, torch.Tensor]] = None,
        mouth_closed: Optional[Dict[str, torch.Tensor]] = None,
        eyes_open: Optional[Dict[str, torch.Tensor]] = None,
        eyes_closed: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor]]:
        """Process all frames into a batch"""

        composer = AnimeTemplateComposer()
        frames = []

        for i in range(frame_count):
            result = composer.compose_frame(
                background_latent=background,
                character_latent=character,
                frame_index=i,
                mouth_open_latent=mouth_open,
                mouth_closed_latent=mouth_closed,
                eyes_open_latent=eyes_open,
                eyes_closed_latent=eyes_closed,
                animation_map=animation_map,
                blend_mode="normal"
            )
            frames.append(result[0]["samples"])

        # Stack all frames into batch
        batched = torch.cat(frames, dim=0)

        return ({"samples": batched},)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AnimeTemplateComposer": AnimeTemplateComposer,
    "AnimeTemplateLoader": AnimeTemplateLoader,
    "AnimationMapGenerator": AnimationMapGenerator,
    "TemplateFrameBatcher": TemplateFrameBatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimeTemplateComposer": "🎨 Anime Template Composer",
    "AnimeTemplateLoader": "📁 Anime Template Loader",
    "AnimationMapGenerator": "🎬 Animation Map Generator",
    "TemplateFrameBatcher": "📦 Template Frame Batcher",
}

# For standalone usage
__all__ = [
    "AnimeTemplateComposer",
    "AnimeTemplateLoader",
    "AnimationMapGenerator",
    "TemplateFrameBatcher",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
