"""
ComfyUI Custom Nodes for Anime Video Generation
"""

from .anime_template_composer import (
    AnimeTemplateComposer,
    AnimeTemplateLoader,
    AnimationMapGenerator,
    TemplateFrameBatcher,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS
)

__all__ = [
    "AnimeTemplateComposer",
    "AnimeTemplateLoader",
    "AnimationMapGenerator",
    "TemplateFrameBatcher",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]

