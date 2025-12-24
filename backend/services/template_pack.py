"""
Template Pack System
User-friendly template pack loading and validation
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from loguru import logger


# JSON Schema for template packs
TEMPLATE_PACK_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "AnimeTemplatePack",
    "type": "object",
    "required": ["name", "version", "resolution", "layers"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Template pack name"
        },
        "version": {
            "type": "string",
            "pattern": "^\\d+\\.\\d+\\.\\d+$",
            "description": "Semantic version"
        },
        "author": {
            "type": "string",
            "description": "Template pack author"
        },
        "description": {
            "type": "string",
            "description": "Template pack description"
        },
        "resolution": {
            "type": "object",
            "required": ["width", "height"],
            "properties": {
                "width": {"type": "integer", "minimum": 64, "maximum": 4096},
                "height": {"type": "integer", "minimum": 64, "maximum": 4096}
            }
        },
        "layers": {
            "type": "object",
            "required": ["background", "character"],
            "properties": {
                "background": {"type": "string"},
                "character": {"type": "string"},
                "mouth_open": {"type": "string"},
                "mouth_closed": {"type": "string"},
                "eyes_open": {"type": "string"},
                "eyes_closed": {"type": "string"},
                "eyes_half": {"type": "string"}
            }
        },
        "animation": {
            "type": "object",
            "properties": {
                "default_mode": {
                    "type": "string",
                    "enum": ["idle", "talking", "emotional"]
                },
                "blink_interval": {"type": "number", "minimum": 0.5},
                "breathing_enabled": {"type": "boolean"},
                "breathing_amplitude": {"type": "number"},
                "sway_enabled": {"type": "boolean"}
            }
        },
        "emotions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "layers": {"type": "object"},
                    "color_tint": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 3,
                        "maxItems": 4
                    }
                }
            }
        },
        "compatibility": {
            "type": "object",
            "properties": {
                "animatediff_version": {"type": "string"},
                "min_comfyui_version": {"type": "string"},
                "sd_version": {"type": "string"}
            }
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}


@dataclass
class TemplatePackMetadata:
    """Template pack metadata structure"""
    name: str
    version: str
    resolution: Dict[str, int]
    layers: Dict[str, str]
    author: str = "Unknown"
    description: str = ""
    animation: Dict[str, Any] = field(default_factory=dict)
    emotions: List[Dict[str, Any]] = field(default_factory=list)
    compatibility: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TemplatePackMetadata':
        return cls(
            name=data.get("name", "Unknown"),
            version=data.get("version", "1.0.0"),
            resolution=data.get("resolution", {"width": 512, "height": 512}),
            layers=data.get("layers", {}),
            author=data.get("author", "Unknown"),
            description=data.get("description", ""),
            animation=data.get("animation", {}),
            emotions=data.get("emotions", []),
            compatibility=data.get("compatibility", {}),
            tags=data.get("tags", [])
        )

    def save(self, filepath: str) -> bool:
        """Save metadata to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False


class TemplatePackValidator:
    """Validate template packs against schema"""

    REQUIRED_LAYERS = ["background", "character"]
    OPTIONAL_LAYERS = ["mouth_open", "mouth_closed", "eyes_open", "eyes_closed", "eyes_half"]
    SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp"]

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, template_dir: str) -> bool:
        """
        Validate a template pack directory

        Args:
            template_dir: Path to template pack directory

        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []

        template_path = Path(template_dir)

        # Check directory exists
        if not template_path.exists():
            self.errors.append(f"Template directory not found: {template_dir}")
            return False

        if not template_path.is_dir():
            self.errors.append(f"Path is not a directory: {template_dir}")
            return False

        # Check for manifest file
        manifest_path = template_path / "anime_template.json"
        if not manifest_path.exists():
            # Try to auto-detect layers
            self.warnings.append("Missing anime_template.json manifest file, using auto-detection")
            return self._validate_auto_detect(template_path)

        # Load and validate manifest
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in manifest: {e}")
            return False

        # Validate manifest structure
        if not self._validate_manifest_structure(manifest):
            return False

        # Validate layer files exist
        if not self._validate_layer_files(template_path, manifest):
            return False

        # Validate resolution consistency
        if not self._validate_resolution(template_path, manifest):
            return False

        # Validate compatibility
        self._check_compatibility(manifest)

        return len(self.errors) == 0

    def _validate_auto_detect(self, template_path: Path) -> bool:
        """Auto-detect and validate layers without manifest"""

        layer_patterns = {
            "background": ["background.png", "bg.png", "back.png"],
            "character": ["character.png", "char.png", "body.png", "main.png"],
            "mouth_open": ["mouth_open.png", "mouth_o.png"],
            "mouth_closed": ["mouth_closed.png", "mouth_c.png"],
            "eyes_open": ["eyes_open.png", "eye_open.png"],
            "eyes_closed": ["eyes_closed.png", "eye_closed.png"],
        }

        found_layers = {}
        for layer_name, patterns in layer_patterns.items():
            for pattern in patterns:
                if (template_path / pattern).exists():
                    found_layers[layer_name] = pattern
                    break

        # Check required layers
        for required in self.REQUIRED_LAYERS:
            if required not in found_layers:
                # Check for any image file
                images = list(template_path.glob("*.png")) + list(template_path.glob("*.jpg"))
                if not images:
                    self.errors.append(f"No image files found in template directory")
                    return False
                else:
                    self.warnings.append(f"Missing {required} layer, using first available image")

        return len(self.errors) == 0

    def _validate_manifest_structure(self, manifest: Dict) -> bool:
        """Validate manifest has required fields"""

        required_fields = ["name", "version", "resolution", "layers"]

        for field in required_fields:
            if field not in manifest:
                self.errors.append(f"Missing required field: {field}")
                return False

        # Validate resolution
        res = manifest.get("resolution", {})
        if "width" not in res or "height" not in res:
            self.errors.append("Resolution must have width and height")
            return False

        # Validate layers
        layers = manifest.get("layers", {})
        for required_layer in self.REQUIRED_LAYERS:
            if required_layer not in layers:
                self.errors.append(f"Missing required layer: {required_layer}")
                return False

        return True

    def _validate_layer_files(self, template_path: Path, manifest: Dict) -> bool:
        """Check all referenced layer files exist"""

        layers = manifest.get("layers", {})
        valid = True

        for layer_name, layer_file in layers.items():
            layer_path = template_path / layer_file

            if not layer_path.exists():
                if layer_name in self.REQUIRED_LAYERS:
                    self.errors.append(f"Required layer file not found: {layer_file}")
                    valid = False
                else:
                    self.warnings.append(f"Optional layer file not found: {layer_file}")
            else:
                # Check file format
                if layer_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                    self.errors.append(
                        f"Unsupported format for {layer_name}: {layer_path.suffix}"
                    )
                    valid = False

        return valid

    def _validate_resolution(self, template_path: Path, manifest: Dict) -> bool:
        """Verify all layers match declared resolution"""
        from PIL import Image

        expected_width = manifest["resolution"]["width"]
        expected_height = manifest["resolution"]["height"]

        layers = manifest.get("layers", {})
        valid = True

        for layer_name, layer_file in layers.items():
            layer_path = template_path / layer_file

            if layer_path.exists():
                try:
                    with Image.open(layer_path) as img:
                        if img.size != (expected_width, expected_height):
                            self.warnings.append(
                                f"Layer {layer_name} has different resolution: "
                                f"{img.size} (expected {expected_width}x{expected_height})"
                            )
                except Exception as e:
                    self.errors.append(f"Failed to read layer {layer_name}: {e}")
                    valid = False

        return valid

    def _check_compatibility(self, manifest: Dict) -> None:
        """Check version compatibility"""

        compat = manifest.get("compatibility", {})

        animatediff_version = compat.get("animatediff_version", "")
        if animatediff_version:
            # Check if version is supported (simplified check)
            try:
                major = int(animatediff_version.split(".")[0])
                if major < 1:
                    self.warnings.append(
                        f"AnimateDiff version {animatediff_version} may not be fully supported"
                    )
            except (ValueError, IndexError):
                pass

    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation errors and warnings"""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "valid": len(self.errors) == 0
        }


class TemplatePackLoader:
    """Load and manage template packs"""

    def __init__(self, packs_directory: str = "templates"):
        self.packs_directory = Path(packs_directory)
        self.validator = TemplatePackValidator()
        self.loaded_packs: Dict[str, TemplatePackMetadata] = {}

    def discover_packs(self) -> List[Dict[str, str]]:
        """Find all template packs in directory"""
        packs = []

        if not self.packs_directory.exists():
            return packs

        for item in self.packs_directory.iterdir():
            if item.is_dir():
                manifest = item / "anime_template.json"
                pack_info = {
                    "path": str(item),
                    "name": item.name,
                    "has_manifest": manifest.exists()
                }

                if manifest.exists():
                    try:
                        with open(manifest, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            pack_info["name"] = data.get("name", item.name)
                            pack_info["version"] = data.get("version", "1.0.0")
                            pack_info["author"] = data.get("author", "Unknown")
                    except:
                        pass

                packs.append(pack_info)

        return packs

    def load_pack(self, pack_path: str) -> Optional[TemplatePackMetadata]:
        """Load a validated template pack"""

        # Validate first
        if not self.validator.validate(pack_path):
            report = self.validator.get_validation_report()
            logger.error(f"Template pack validation failed: {report['errors']}")
            return None

        # Load manifest
        manifest_path = Path(pack_path) / "anime_template.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        else:
            # Create default manifest from auto-detection
            manifest = self._create_default_manifest(pack_path)

        metadata = TemplatePackMetadata.from_dict(manifest)
        self.loaded_packs[pack_path] = metadata

        logger.info(f"✅ Loaded template pack: {metadata.name} v{metadata.version}")

        return metadata

    def _create_default_manifest(self, pack_path: str) -> Dict:
        """Create default manifest from auto-detected layers"""
        path = Path(pack_path)

        layers = {}
        layer_patterns = {
            "background": ["background.png", "bg.png"],
            "character": ["character.png", "char.png"],
            "mouth_open": ["mouth_open.png"],
            "mouth_closed": ["mouth_closed.png"],
            "eyes_open": ["eyes_open.png"],
            "eyes_closed": ["eyes_closed.png"],
        }

        for layer_name, patterns in layer_patterns.items():
            for pattern in patterns:
                if (path / pattern).exists():
                    layers[layer_name] = pattern
                    break

        return {
            "name": path.name,
            "version": "1.0.0",
            "resolution": {"width": 512, "height": 512},
            "layers": layers,
            "animation": {
                "default_mode": "idle",
                "blink_interval": 3.0,
                "breathing_enabled": True
            }
        }

    def get_pack_layers(self, pack_path: str) -> Dict[str, Path]:
        """Get absolute paths to all layer files"""

        if pack_path not in self.loaded_packs:
            self.load_pack(pack_path)

        metadata = self.loaded_packs.get(pack_path)
        if not metadata:
            return {}

        pack_dir = Path(pack_path)
        return {
            layer_name: pack_dir / layer_file
            for layer_name, layer_file in metadata.layers.items()
        }

    def create_pack_template(self, output_path: str, name: str) -> str:
        """Create a template pack directory structure"""

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "effects").mkdir(exist_ok=True)
        (output_dir / "emotions").mkdir(exist_ok=True)

        # Create manifest template
        manifest = {
            "name": name,
            "version": "1.0.0",
            "author": "Your Name",
            "description": "Anime character template pack",
            "resolution": {
                "width": 512,
                "height": 512
            },
            "layers": {
                "background": "background.png",
                "character": "character.png",
                "mouth_open": "mouth_open.png",
                "mouth_closed": "mouth_closed.png",
                "eyes_open": "eyes_open.png",
                "eyes_closed": "eyes_closed.png"
            },
            "animation": {
                "default_mode": "idle",
                "blink_interval": 3.0,
                "breathing_enabled": True,
                "breathing_amplitude": 0.02,
                "sway_enabled": True
            },
            "emotions": [
                {
                    "name": "happy",
                    "layers": {
                        "character": "emotions/happy.png"
                    }
                },
                {
                    "name": "sad",
                    "layers": {
                        "character": "emotions/sad.png"
                    }
                }
            ],
            "compatibility": {
                "animatediff_version": "1.0.0",
                "min_comfyui_version": "0.1.0",
                "sd_version": "1.5"
            },
            "tags": ["anime", "character", "template"]
        }

        manifest_path = output_dir / "anime_template.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        # Create placeholder images
        from PIL import Image

        placeholder_layers = [
            "background.png", "character.png",
            "mouth_open.png", "mouth_closed.png",
            "eyes_open.png", "eyes_closed.png"
        ]

        for layer_file in placeholder_layers:
            img = Image.new('RGBA', (512, 512), (128, 128, 128, 128))
            img.save(output_dir / layer_file)

        logger.info(f"✅ Created template pack structure at: {output_path}")

        return str(output_dir)


def get_template_pack_schema() -> Dict:
    """Return the JSON schema for template packs"""
    return TEMPLATE_PACK_SCHEMA


def validate_template_pack(pack_path: str) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate a template pack

    Returns:
        (is_valid, errors, warnings)
    """
    validator = TemplatePackValidator()
    is_valid = validator.validate(pack_path)
    report = validator.get_validation_report()
    return is_valid, report["errors"], report["warnings"]

