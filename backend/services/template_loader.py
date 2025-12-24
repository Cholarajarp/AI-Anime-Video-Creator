"""
Template Loader - PNG/Template Ingestion Engine
================================================
Converts static PNG templates into animation-ready inputs.
Supports layered anime character templates with:
- Background, character, mouth states, eye states
- Optional alpha-mask extraction for compositing
- Animation mode selection (idle, talking, emotional)
"""

import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np
from PIL import Image
import torch
from loguru import logger


class TemplateLoader:
    """
    Load and process anime character templates for animation.
    Converts PNG layers into tensors for video generation.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.cache = {}
        logger.info(f"TemplateLoader initialized on {device}")

    def load_template_pack(
        self,
        template_dir: str,
        target_resolution: Tuple[int, int] = (512, 512),
        animation_mode: str = "idle"
    ) -> Dict[str, torch.Tensor]:
        """
        Load a complete template pack from directory.

        Expected directory structure:
            /template/
                background.png
                character.png
                mouth_open.png
                mouth_closed.png
                eyes_open.png
                eyes_closed.png
                effects/*.png

        Args:
            template_dir: Path to template directory
            target_resolution: (width, height) to normalize all images
            animation_mode: One of ["idle", "talking", "emotional"]

        Returns:
            Dictionary mapping layer names to tensors
        """
        template_path = Path(template_dir)

        if not template_path.exists():
            logger.warning(f"Template directory not found: {template_dir}")
            return self._create_default_template(target_resolution)

        result = {}

        # Core layers
        layer_files = {
            "background": ["background.png", "bg.png"],
            "character": ["character.png", "char.png", "body.png"],
            "mouth_open": ["mouth_open.png", "mouth_o.png"],
            "mouth_closed": ["mouth_closed.png", "mouth_c.png"],
            "eyes_open": ["eyes_open.png", "eye_open.png"],
            "eyes_closed": ["eyes_closed.png", "eye_closed.png"],
            "eyes_half": ["eyes_half.png", "eye_half.png"],
        }

        for layer_name, possible_files in layer_files.items():
            for filename in possible_files:
                filepath = template_path / filename
                if filepath.exists():
                    tensor, alpha = self._load_image_as_tensor(
                        str(filepath), target_resolution
                    )
                    result[layer_name] = tensor
                    result[f"{layer_name}_alpha"] = alpha
                    logger.debug(f"Loaded layer: {layer_name}")
                    break

        # Load effect layers
        effects_dir = template_path / "effects"
        if effects_dir.exists():
            result["effects"] = []
            for effect_file in effects_dir.glob("*.png"):
                tensor, alpha = self._load_image_as_tensor(
                    str(effect_file), target_resolution
                )
                result["effects"].append({
                    "name": effect_file.stem,
                    "tensor": tensor,
                    "alpha": alpha
                })

        # Add animation mode metadata
        result["animation_mode"] = animation_mode
        result["resolution"] = target_resolution

        # Generate default layers if missing
        result = self._fill_missing_layers(result, target_resolution)

        logger.info(f"Template loaded: {len(result)} layers")
        return result

    def _load_image_as_tensor(
        self,
        filepath: str,
        target_resolution: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image and convert to normalized tensor.

        Returns:
            (image_tensor, alpha_tensor) - both normalized to [0, 1]
        """
        img = Image.open(filepath)

        # Convert to RGBA if needed
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Resize to target resolution
        img = img.resize(target_resolution, Image.Resampling.LANCZOS)

        # Convert to numpy
        np_img = np.array(img).astype(np.float32) / 255.0

        # Separate RGB and Alpha
        rgb = np_img[:, :, :3]
        alpha = np_img[:, :, 3:4]

        # Convert to tensors [C, H, W]
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)
        alpha_tensor = torch.from_numpy(alpha).permute(2, 0, 1).to(self.device)

        return rgb_tensor, alpha_tensor

    def _create_default_template(
        self,
        resolution: Tuple[int, int]
    ) -> Dict[str, torch.Tensor]:
        """Create a default template when no files found."""
        w, h = resolution

        # Default background gradient
        bg = torch.zeros(3, h, w, device=self.device)
        for i in range(h):
            t = i / h
            bg[0, i, :] = 0.2 + t * 0.1  # R
            bg[1, i, :] = 0.2 + t * 0.15  # G
            bg[2, i, :] = 0.3 + t * 0.2  # B

        return {
            "background": bg,
            "background_alpha": torch.ones(1, h, w, device=self.device),
            "character": torch.zeros(3, h, w, device=self.device),
            "character_alpha": torch.zeros(1, h, w, device=self.device),
            "animation_mode": "idle",
            "resolution": resolution
        }

    def _fill_missing_layers(
        self,
        result: Dict,
        resolution: Tuple[int, int]
    ) -> Dict:
        """Fill in missing layers with defaults."""
        w, h = resolution
        empty = torch.zeros(3, h, w, device=self.device)
        empty_alpha = torch.zeros(1, h, w, device=self.device)

        required = ["background", "character", "mouth_open", "mouth_closed",
                   "eyes_open", "eyes_closed"]

        for layer in required:
            if layer not in result:
                result[layer] = empty.clone()
                result[f"{layer}_alpha"] = empty_alpha.clone()

        return result

    def composite_frame(
        self,
        template: Dict[str, torch.Tensor],
        mouth_state: str = "closed",  # "open" or "closed"
        eye_state: str = "open",  # "open", "closed", "half"
        effect_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Composite template layers into a single frame.

        Args:
            template: Loaded template dictionary
            mouth_state: Current mouth state
            eye_state: Current eye state
            effect_names: List of effect layer names to include

        Returns:
            Composited frame tensor [3, H, W]
        """
        # Start with background
        frame = template["background"].clone()

        # Helper for alpha compositing
        def blend(base, layer, alpha):
            return base * (1 - alpha) + layer * alpha

        # Add character
        if "character" in template:
            alpha = template.get("character_alpha", torch.ones_like(template["character"][:1]))
            frame = blend(frame, template["character"], alpha)

        # Add eyes
        eye_layer = f"eyes_{eye_state}"
        if eye_layer in template:
            alpha = template.get(f"{eye_layer}_alpha", torch.ones_like(template[eye_layer][:1]))
            frame = blend(frame, template[eye_layer], alpha)

        # Add mouth
        mouth_layer = f"mouth_{mouth_state}"
        if mouth_layer in template:
            alpha = template.get(f"{mouth_layer}_alpha", torch.ones_like(template[mouth_layer][:1]))
            frame = blend(frame, template[mouth_layer], alpha)

        # Add effects
        if effect_names and "effects" in template:
            for effect in template["effects"]:
                if effect["name"] in effect_names:
                    alpha = effect["alpha"]
                    frame = blend(frame, effect["tensor"], alpha)

        return frame

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor [3, H, W] to PIL Image."""
        np_img = tensor.cpu().numpy().transpose(1, 2, 0)
        np_img = (np_img * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(np_img)

    def pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor [3, H, W]."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        np_img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(np_img).permute(2, 0, 1).to(self.device)


class TemplatePack:
    """
    Represents a validated template pack with metadata.
    """

    def __init__(self, pack_path: str):
        self.path = Path(pack_path)
        self.metadata = self._load_metadata()
        self.valid = self._validate()

    def _load_metadata(self) -> Dict:
        """Load template pack metadata."""
        import json

        meta_file = self.path / "anime_template.json"
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Generate default metadata
        return {
            "name": self.path.name,
            "version": "1.0.0",
            "resolution": [512, 512],
            "author": "Unknown",
            "description": "Anime character template",
            "layers": [],
            "emotions": ["neutral"],
            "compatible_version": "1.0.0"
        }

    def _validate(self) -> bool:
        """Validate template pack structure."""
        required_layers = ["character.png", "background.png"]

        for layer in required_layers:
            if not (self.path / layer).exists():
                # Check alternate names
                alt_found = False
                for alt in [layer.replace(".png", "_base.png"), layer.replace(".png", "_main.png")]:
                    if (self.path / alt).exists():
                        alt_found = True
                        break
                if not alt_found:
                    logger.warning(f"Missing required layer: {layer}")

        return True

    @property
    def name(self) -> str:
        return self.metadata.get("name", "Unknown")

    @property
    def resolution(self) -> Tuple[int, int]:
        res = self.metadata.get("resolution", [512, 512])
        return (res[0], res[1])

    @property
    def emotions(self) -> List[str]:
        return self.metadata.get("emotions", ["neutral"])


# JSON Schema for template packs
TEMPLATE_PACK_SCHEMA = {
    "type": "object",
    "required": ["name", "version", "resolution"],
    "properties": {
        "name": {"type": "string", "description": "Template pack name"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "resolution": {
            "type": "array",
            "items": {"type": "integer", "minimum": 128, "maximum": 2048},
            "minItems": 2,
            "maxItems": 2
        },
        "author": {"type": "string"},
        "description": {"type": "string"},
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "file": {"type": "string"},
                    "z_order": {"type": "integer"}
                }
            }
        },
        "emotions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "animations": {
            "type": "object",
            "properties": {
                "idle": {"type": "object"},
                "talking": {"type": "object"},
                "emotional": {"type": "object"}
            }
        },
        "compatible_version": {"type": "string"}
    }
}


def validate_template_pack(pack_path: str) -> Tuple[bool, List[str]]:
    """
    Validate a template pack against the schema.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    pack = Path(pack_path)

    if not pack.exists():
        return False, ["Template pack directory does not exist"]

    # Check for metadata file
    meta_file = pack / "anime_template.json"
    if meta_file.exists():
        import json
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Basic validation
            if "name" not in metadata:
                errors.append("Missing 'name' in metadata")
            if "version" not in metadata:
                errors.append("Missing 'version' in metadata")
            if "resolution" not in metadata:
                errors.append("Missing 'resolution' in metadata")
            elif len(metadata["resolution"]) != 2:
                errors.append("Resolution must be [width, height]")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in metadata: {e}")

    # Check for required image files
    has_character = any((pack / f).exists() for f in ["character.png", "char.png", "body.png"])
    if not has_character:
        errors.append("Missing character layer (character.png)")

    is_valid = len(errors) == 0
    return is_valid, errors

