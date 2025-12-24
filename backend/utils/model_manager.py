"""
Model Manager Utility
=====================
Utilities for discovering and managing AI models.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ModelInfo:
    """Information about an AI model file."""
    name: str
    filename: str
    path: Path
    type: str
    size_mb: float
    description: Optional[str] = None


class ModelManager:
    """
    Manages discovery and validation of AI models.

    Supports:
    - Checkpoint models (.safetensors, .ckpt)
    - Motion modules (.ckpt, .pth)
    - LoRA models (.safetensors)
    - VAE models (.safetensors, .pt)
    """

    SUPPORTED_EXTENSIONS = {
        "checkpoint": [".safetensors", ".ckpt"],
        "motion_module": [".ckpt", ".pth"],
        "lora": [".safetensors"],
        "vae": [".safetensors", ".pt"]
    }

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._cache: Dict[str, List[ModelInfo]] = {}

    def _scan_directory(
        self,
        directory: Path,
        model_type: str
    ) -> List[ModelInfo]:
        """Scan a directory for model files."""
        models = []

        if not directory.exists():
            logger.warning(f"Model directory not found: {directory}")
            return models

        extensions = self.SUPPORTED_EXTENSIONS.get(model_type, [])

        for ext in extensions:
            for file_path in directory.glob(f"*{ext}"):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    models.append(ModelInfo(
                        name=file_path.stem,
                        filename=file_path.name,
                        path=file_path,
                        type=model_type,
                        size_mb=size_mb
                    ))

        return sorted(models, key=lambda m: m.name.lower())

    def get_checkpoints(self, refresh: bool = False) -> List[ModelInfo]:
        """Get list of available checkpoint models."""
        if "checkpoint" not in self._cache or refresh:
            self._cache["checkpoint"] = self._scan_directory(
                self.models_dir / "checkpoints",
                "checkpoint"
            )
        return self._cache["checkpoint"]

    def get_motion_modules(self, refresh: bool = False) -> List[ModelInfo]:
        """Get list of available motion modules."""
        if "motion_module" not in self._cache or refresh:
            self._cache["motion_module"] = self._scan_directory(
                self.models_dir / "motion_modules",
                "motion_module"
            )
        return self._cache["motion_module"]

    def get_loras(self, refresh: bool = False) -> List[ModelInfo]:
        """Get list of available LoRA models."""
        if "lora" not in self._cache or refresh:
            self._cache["lora"] = self._scan_directory(
                self.models_dir / "loras",
                "lora"
            )
        return self._cache["lora"]

    def get_vaes(self, refresh: bool = False) -> List[ModelInfo]:
        """Get list of available VAE models."""
        if "vae" not in self._cache or refresh:
            self._cache["vae"] = self._scan_directory(
                self.models_dir / "vae",
                "vae"
            )
        return self._cache["vae"]

    def get_all_models(self, refresh: bool = False) -> Dict[str, List[ModelInfo]]:
        """Get all available models organized by type."""
        return {
            "checkpoints": self.get_checkpoints(refresh),
            "motion_modules": self.get_motion_modules(refresh),
            "loras": self.get_loras(refresh),
            "vaes": self.get_vaes(refresh)
        }

    def get_checkpoint_choices(self) -> List[str]:
        """Get checkpoint filenames for dropdown."""
        return [m.filename for m in self.get_checkpoints()]

    def get_motion_module_choices(self) -> List[str]:
        """Get motion module filenames for dropdown."""
        return [m.filename for m in self.get_motion_modules()]

    def get_lora_choices(self) -> List[str]:
        """Get LoRA filenames for dropdown."""
        return [m.filename for m in self.get_loras()]

    def validate_checkpoint(self, filename: str) -> bool:
        """Check if a checkpoint file exists."""
        checkpoints = self.get_checkpoints()
        return any(m.filename == filename for m in checkpoints)

    def validate_motion_module(self, filename: str) -> bool:
        """Check if a motion module file exists."""
        modules = self.get_motion_modules()
        return any(m.filename == filename for m in modules)

    def get_model_path(self, filename: str, model_type: str) -> Optional[Path]:
        """Get the full path to a model file."""
        type_dirs = {
            "checkpoint": "checkpoints",
            "motion_module": "motion_modules",
            "lora": "loras",
            "vae": "vae"
        }

        dir_name = type_dirs.get(model_type)
        if not dir_name:
            return None

        path = self.models_dir / dir_name / filename
        return path if path.exists() else None

    def get_total_size(self) -> float:
        """Get total size of all models in GB."""
        total_mb = 0.0
        for models in self.get_all_models().values():
            total_mb += sum(m.size_mb for m in models)
        return total_mb / 1024

    def refresh_cache(self):
        """Force refresh of model cache."""
        self._cache = {}
        self.get_all_models(refresh=True)
        logger.info("Model cache refreshed")


# Recommended models with download info
RECOMMENDED_MODELS = {
    "checkpoints": [
        {
            "name": "DreamShaper 8",
            "filename": "dreamshaper_8.safetensors",
            "url": "https://huggingface.co/Lykon/dreamshaper-8/resolve/main/dreamshaper_8.safetensors",
            "description": "Semi-realistic anime style, high detail"
        },
        {
            "name": "MeinaMix V11",
            "filename": "meinamix_v11.safetensors",
            "url": "https://civitai.com/api/download/models/119057",
            "description": "Pure 2D anime style, vibrant colors"
        },
        {
            "name": "Counterfeit V3",
            "filename": "counterfeit_v3.safetensors",
            "url": "https://huggingface.co/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0.safetensors",
            "description": "High-quality illustration style"
        }
    ],
    "motion_modules": [
        {
            "name": "AnimateDiff v2",
            "filename": "mm_sd_v15_v2.ckpt",
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
            "description": "Standard motion module for SD 1.5"
        },
        {
            "name": "AnimateDiff v3",
            "filename": "v3_sd15_mm.ckpt",
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt",
            "description": "Enhanced motion quality"
        }
    ]
}


def get_recommended_models() -> Dict[str, List[Dict[str, str]]]:
    """Get list of recommended models with download URLs."""
    return RECOMMENDED_MODELS

