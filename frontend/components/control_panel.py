"""
Control Panel Component
=======================
Main control panel with prompt inputs, voice selection, and generation settings.
"""

import gradio as gr
from typing import List, Tuple, Optional, Callable


# Default prompt templates
PROMPT_TEMPLATES = {
    "Anime Girl - Portrait": "1girl, solo, anime, looking at viewer, smile, beautiful face, detailed eyes, school uniform, cherry blossoms, soft lighting",
    "Anime Boy - Dynamic": "1boy, solo, anime, dynamic pose, cool expression, spiky hair, jacket, urban background, dramatic lighting",
    "Cute Chibi": "chibi, 1girl, kawaii, big eyes, pastel colors, simple background, happy expression, bouncing",
    "Fantasy Scene": "anime, fantasy, magical forest, glowing particles, mystical atmosphere, ethereal lighting, detailed background",
    "Action Pose": "anime, dynamic action pose, motion blur, intense expression, energy effects, dramatic angle",
}

NEGATIVE_PROMPT_PRESETS = {
    "Standard": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    "Strict": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed, mutated, disfigured, ugly, duplicate, morbid, mutation",
    "Minimal": "lowres, bad quality, watermark, text, signature",
}


def create_control_panel(
    voice_choices: List[Tuple[str, str]],
    checkpoint_choices: List[str],
    motion_module_choices: List[str]
) -> dict:
    """
    Create the main control panel component.

    Args:
        voice_choices: List of (display_name, voice_id) tuples
        checkpoint_choices: List of available checkpoint filenames
        motion_module_choices: List of available motion module filenames

    Returns:
        Dictionary of all control components
    """
    with gr.Column(elem_classes=["control-panel"]) as panel:
        # Header
        gr.Markdown(
            """
            ## 🎨 Generation Controls
            Configure your anime video generation settings below.
            """,
            elem_classes=["panel-header"]
        )

        # Prompt Section
        with gr.Accordion("📝 Prompts", open=True):
            with gr.Row():
                prompt_template = gr.Dropdown(
                    choices=list(PROMPT_TEMPLATES.keys()),
                    label="Quick Templates",
                    value=None,
                    interactive=True,
                    elem_classes=["template-dropdown"]
                )
                apply_template_btn = gr.Button(
                    "Apply",
                    size="sm",
                    variant="secondary"
                )

            prompt = gr.Textbox(
                label="Visual Prompt",
                placeholder="Describe the visual content of your video...\nExample: 1girl, anime, smile, waving hand, pink hair, cute outfit",
                lines=4,
                max_lines=8,
                elem_classes=["prompt-input"]
            )

            with gr.Accordion("Negative Prompt", open=False):
                negative_preset = gr.Dropdown(
                    choices=list(NEGATIVE_PROMPT_PRESETS.keys()),
                    label="Preset",
                    value="Standard",
                    interactive=True
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value=NEGATIVE_PROMPT_PRESETS["Standard"],
                    lines=3,
                    max_lines=6,
                    elem_classes=["negative-prompt-input"]
                )

        # Audio Script Section
        with gr.Accordion("🎤 Audio Script", open=True):
            script = gr.Textbox(
                label="Script / Dialogue",
                placeholder="Enter the text that will be converted to speech...\nThis determines the length of your video.",
                lines=4,
                max_lines=10,
                elem_classes=["script-input"]
            )

            with gr.Row():
                voice = gr.Dropdown(
                    choices=voice_choices,
                    label="Voice",
                    value=voice_choices[0][1] if voice_choices else None,
                    interactive=True,
                    elem_classes=["voice-dropdown"]
                )
                preview_voice_btn = gr.Button(
                    "🔊 Preview",
                    size="sm",
                    variant="secondary",
                    elem_classes=["preview-btn"]
                )

            with gr.Row():
                speech_rate = gr.Slider(
                    minimum=-50,
                    maximum=50,
                    value=0,
                    step=5,
                    label="Speech Rate (%)",
                    info="Adjust speaking speed"
                )
                speech_pitch = gr.Slider(
                    minimum=-20,
                    maximum=20,
                    value=0,
                    step=2,
                    label="Pitch (Hz)",
                    info="Adjust voice pitch"
                )

            audio_preview = gr.Audio(
                label="Voice Preview",
                visible=False,
                elem_classes=["audio-preview"]
            )

        # Model Settings Section
        with gr.Accordion("🤖 Model Settings", open=False):
            checkpoint = gr.Dropdown(
                choices=checkpoint_choices,
                label="Checkpoint Model",
                value=checkpoint_choices[0] if checkpoint_choices else None,
                interactive=True,
                info="Base anime model for generation"
            )

            motion_module = gr.Dropdown(
                choices=motion_module_choices,
                label="Motion Module",
                value=motion_module_choices[0] if motion_module_choices else None,
                interactive=True,
                info="AnimateDiff motion model"
            )

        # Generation Settings Section
        with gr.Accordion("⚙️ Generation Settings", open=False):
            with gr.Row():
                width = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Width",
                    info="Video width in pixels"
                )
                height = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Height",
                    info="Video height in pixels"
                )

            with gr.Row():
                fps = gr.Slider(
                    minimum=8,
                    maximum=30,
                    value=15,
                    step=1,
                    label="FPS",
                    info="Frames per second"
                )
                steps = gr.Slider(
                    minimum=10,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Steps",
                    info="Diffusion steps (higher = better quality, slower)"
                )

            with gr.Row():
                cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.0,
                    step=0.5,
                    label="CFG Scale",
                    info="Prompt adherence (higher = stricter)"
                )
                seed = gr.Number(
                    value=-1,
                    label="Seed",
                    info="-1 for random",
                    precision=0
                )

            randomize_seed = gr.Checkbox(
                label="Randomize seed on each generation",
                value=True
            )

        # Quality Presets
        with gr.Accordion("🎯 Quality Presets", open=False):
            quality_preset = gr.Radio(
                choices=["Draft", "Standard", "High Quality"],
                value="Standard",
                label="Quality Preset",
                info="Quickly set quality-related parameters"
            )

            preset_info = gr.Markdown(
                """
                **Standard**: 512x512, 20 steps, CFG 7.0 - Balanced quality and speed
                """,
                elem_classes=["preset-info"]
            )

        # Priority Selection
        with gr.Row():
            priority = gr.Radio(
                choices=["high", "normal", "low"],
                value="normal",
                label="Job Priority",
                info="Higher priority jobs are processed first"
            )

        # Action Buttons
        with gr.Row(elem_classes=["action-buttons"]):
            generate_btn = gr.Button(
                "🎬 Generate Video",
                variant="primary",
                size="lg",
                elem_classes=["generate-btn"]
            )

            clear_btn = gr.Button(
                "🗑️ Clear",
                variant="secondary",
                size="lg",
                elem_classes=["clear-btn"]
            )

    # Return all components
    return {
        "panel": panel,
        "prompt": prompt,
        "prompt_template": prompt_template,
        "apply_template_btn": apply_template_btn,
        "negative_prompt": negative_prompt,
        "negative_preset": negative_preset,
        "script": script,
        "voice": voice,
        "preview_voice_btn": preview_voice_btn,
        "speech_rate": speech_rate,
        "speech_pitch": speech_pitch,
        "audio_preview": audio_preview,
        "checkpoint": checkpoint,
        "motion_module": motion_module,
        "width": width,
        "height": height,
        "fps": fps,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "quality_preset": quality_preset,
        "preset_info": preset_info,
        "priority": priority,
        "generate_btn": generate_btn,
        "clear_btn": clear_btn
    }


def apply_template(template_name: str) -> str:
    """Apply a prompt template."""
    return PROMPT_TEMPLATES.get(template_name, "")


def apply_negative_preset(preset_name: str) -> str:
    """Apply a negative prompt preset."""
    return NEGATIVE_PROMPT_PRESETS.get(preset_name, NEGATIVE_PROMPT_PRESETS["Standard"])


def apply_quality_preset(preset: str) -> Tuple[int, int, int, float, str]:
    """
    Apply a quality preset.

    Returns:
        Tuple of (width, height, steps, cfg_scale, info_text)
    """
    presets = {
        "Draft": {
            "width": 384,
            "height": 384,
            "steps": 15,
            "cfg_scale": 6.0,
            "info": "**Draft**: 384x384, 15 steps, CFG 6.0 - Fast preview quality"
        },
        "Standard": {
            "width": 512,
            "height": 512,
            "steps": 20,
            "cfg_scale": 7.0,
            "info": "**Standard**: 512x512, 20 steps, CFG 7.0 - Balanced quality and speed"
        },
        "High Quality": {
            "width": 768,
            "height": 768,
            "steps": 30,
            "cfg_scale": 7.5,
            "info": "**High Quality**: 768x768, 30 steps, CFG 7.5 - Best quality, slower"
        }
    }

    p = presets.get(preset, presets["Standard"])
    return p["width"], p["height"], p["steps"], p["cfg_scale"], p["info"]


def format_speech_rate(value: int) -> str:
    """Format speech rate value for Edge-TTS."""
    if value >= 0:
        return f"+{value}%"
    return f"{value}%"


def format_speech_pitch(value: int) -> str:
    """Format speech pitch value for Edge-TTS."""
    if value >= 0:
        return f"+{value}Hz"
    return f"{value}Hz"


def clear_inputs() -> Tuple:
    """Clear all input fields."""
    return (
        "",  # prompt
        NEGATIVE_PROMPT_PRESETS["Standard"],  # negative_prompt
        "",  # script
        0,   # speech_rate
        0,   # speech_pitch
        512, # width
        512, # height
        15,  # fps
        20,  # steps
        7.0, # cfg_scale
        -1,  # seed
        True, # randomize_seed
        "Standard",  # quality_preset
        "normal"  # priority
    )

