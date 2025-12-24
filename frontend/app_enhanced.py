"""
AI Video Creator - Enhanced Application with Advanced Features
===============================================================
Next-Generation Anime Video Synthesis Platform
Supports: Japanese, English, Hindi, Kannada, Korean, Chinese
"""

import gradio as gr
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

from backend.services.video_generator import VideoGenerator

# Initialize video generator
video_gen = VideoGenerator()

# ============================================================================
# Enhanced Configuration
# ============================================================================

VOICE_CHOICES = [
    ("🇯🇵 Nanami (Japanese Female) - Energetic Anime", "ja-JP-NanamiNeural"),
    ("🇯🇵 Aoi (Japanese Female) - Gentle & Soft", "ja-JP-AoiNeural"),
    ("🇯🇵 Keita (Japanese Male) - Young & Dynamic", "ja-JP-KeitaNeural"),
    ("🇯🇵 Daichi (Japanese Male) - Deep & Confident", "ja-JP-DaichiNeural"),
    ("🇺🇸 Aria (English Female) - Narrator", "en-US-AriaNeural"),
    ("🇺🇸 Jenny (English Female) - Friendly", "en-US-JennyNeural"),
    ("🇺🇸 Guy (English Male) - Professional", "en-US-GuyNeural"),
    ("🇰🇷 SunHi (Korean Female) - K-Drama", "ko-KR-SunHiNeural"),
    ("🇨🇳 Xiaoxiao (Chinese Female) - Expressive", "zh-CN-XiaoxiaoNeural"),
    ("🇮🇳 Swara (Hindi Female) - Warm & Expressive", "hi-IN-SwaraNeural"),
    ("🇮🇳 Madhur (Hindi Male) - Professional", "hi-IN-MadhurNeural"),
    ("🇮🇳 Sapna (Kannada Female) - Melodic", "kn-IN-SapnaNeural"),
    ("🇮🇳 Gagan (Kannada Male) - Deep Voice", "kn-IN-GaganNeural"),
]

PROMPT_TEMPLATES = {
    "Anime Girl Portrait 🎀": "1girl, solo, anime, beautiful detailed eyes, long flowing hair, gentle smile, cherry blossoms, soft pastel colors, dreamy atmosphere, high quality, masterpiece",
    "Anime Boy Action 💥": "1boy, solo, anime, dynamic action pose, determined expression, cool outfit, energy effects, dramatic lighting, detailed background, high quality",
    "Cute Chibi Character 🧸": "chibi, kawaii, big sparkling eyes, happy expression, pastel colors, simple background, adorable, high quality",
    "Fantasy Magic Scene ✨": "anime, fantasy landscape, magical particles, glowing effects, mystical atmosphere, ethereal lighting, detailed scenery, enchanted forest",
    "Cyberpunk City 🌃": "anime, cyberpunk city, neon lights, futuristic buildings, rain, night scene, purple and blue tones, atmospheric",
    "School Life 🏫": "anime, school uniform, classroom, cheerful atmosphere, bright lighting, slice of life, detailed background",
    "Romantic Sunset 💕": "anime, couple, sunset sky, warm colors, romantic atmosphere, silhouette, emotional, beautiful scenery",
    "Epic Battle ⚔️": "anime, battle scene, dramatic pose, energy aura, intense expression, action effects, dynamic composition",
}

STYLE_PRESETS = {
    "Anime - Vibrant": {
        "checkpoint": "dreamshaper_8.safetensors",
        "cfg_scale": 7.5,
        "prompt_suffix": ", vivid colors, vibrant, high saturation"
    },
    "Anime - Soft Pastel": {
        "checkpoint": "dreamshaper_8.safetensors",
        "cfg_scale": 7.0,
        "prompt_suffix": ", soft pastel colors, gentle lighting, watercolor style"
    },
    "Anime - Dramatic": {
        "checkpoint": "dreamshaper_8.safetensors",
        "cfg_scale": 8.0,
        "prompt_suffix": ", dramatic lighting, high contrast, cinematic"
    },
    "Realistic Anime": {
        "checkpoint": "dreamshaper_8.safetensors",
        "cfg_scale": 7.0,
        "prompt_suffix": ", semi-realistic, detailed textures, photorealistic anime"
    },
}

NEGATIVE_PROMPT_DEFAULT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, distorted, disfigured, ugly"

CHECKPOINT_CHOICES = [
    "dreamshaper_8.safetensors",
    "meinamix_v11.safetensors",
    "counterfeit_v3.safetensors",
    "anything_v5.safetensors"
]

MOTION_MODULE_CHOICES = [
    "mm_sd_v15_v2.ckpt",
    "mm_sd_v15.ckpt"
]

# ============================================================================
# Global State
# ============================================================================

log_messages = []
generation_history = []

def add_log(message: str, level: str = "info"):
    """Add log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_messages.append({
        "time": timestamp,
        "level": level,
        "message": message
    })

    # Keep only last 50 messages
    if len(log_messages) > 50:
        log_messages.pop(0)

    # Log to logger
    if level == "error":
        logger.error(message)
    elif level == "warning":
        logger.warning(message)
    else:
        logger.info(message)

def get_logs_html():
    """Generate HTML for logs."""
    if not log_messages:
        return "<div style='color: #888; text-align: center; padding: 20px;'>Waiting for activity...</div>"

    html = "<div style='font-family: monospace; font-size: 12px; line-height: 1.6;'>"

    for log in reversed(log_messages[-20:]):  # Show last 20
        color = {
            "error": "#ff6b6b",
            "warning": "#ffd43b",
            "success": "#51cf66",
            "info": "#74c0fc"
        }.get(log["level"], "#adb5bd")

        icon = {
            "error": "❌",
            "warning": "⚠️",
            "success": "✅",
            "info": "ℹ️"
        }.get(log["level"], "•")

        html += f"""
        <div style='margin: 4px 0; color: {color};'>
            <span style='color: #868e96;'>[{log['time']}]</span>
            {icon} {log['message']}
        </div>
        """

    html += "</div>"
    return html

# ============================================================================
# Audio Generation
# ============================================================================

def generate_audio(text: str, voice_id: str, rate: int = 0, pitch: int = 0):
    """Generate audio using Edge-TTS."""
    import uuid
    from pathlib import Path

    if not text or not text.strip():
        add_log("Audio error: Empty text provided", "error")
        return None

    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    output_path = temp_dir / f"audio_{uuid.uuid4().hex[:8]}.mp3"

    try:
        import edge_tts

        rate_str = f"+{rate}%" if rate >= 0 else f"{rate}%"
        pitch_str = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

        async def _generate():
            communicate = edge_tts.Communicate(
                text=text.strip(),
                voice=voice_id,
                rate=rate_str,
                pitch=pitch_str
            )
            await communicate.save(str(output_path))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate())
        loop.close()

        if output_path.exists() and output_path.stat().st_size > 0:
            add_log(f"✅ Audio generated: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)", "success")
            return str(output_path)
        else:
            add_log("Audio generation failed: Empty file", "error")
            return None

    except ImportError:
        add_log("edge-tts not installed. Installing...", "warning")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "edge-tts"])
        return generate_audio(text, voice_id, rate, pitch)
    except Exception as e:
        add_log(f"Audio error: {str(e)}", "error")
        import traceback
        logger.error(traceback.format_exc())
        return None

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    import math

    try:
        # Try mutagen (pure Python)
        try:
            from mutagen.mp3 import MP3
            audio = MP3(audio_path)
            return audio.info.length
        except ImportError:
            # Install mutagen
            add_log("Installing mutagen for audio analysis...", "info")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mutagen"])
            from mutagen.mp3 import MP3
            audio = MP3(audio_path)
            return audio.info.length
    except Exception as e:
        logger.error(f"Duration calculation failed: {e}")
        # Estimate from file size
        import os
        file_size = os.path.getsize(audio_path)
        return file_size / (16 * 1024)

def preview_voice(text: str, voice_id: str, rate: int, pitch: int):
    """Generate voice preview."""
    if not text.strip():
        return None, get_logs_html()

    preview_text = text[:200] if len(text) > 200 else text
    add_log(f"🎤 Generating voice preview: {voice_id}")

    audio_path = generate_audio(preview_text, voice_id, rate, pitch)
    return audio_path, get_logs_html()

# ============================================================================
# Video Generation
# ============================================================================

def on_generate(
    prompt: str,
    negative_prompt: str,
    script: str,
    voice: str,
    rate: int,
    pitch: int,
    style_preset: str,
    checkpoint: str,
    motion_module: str,
    width: int,
    height: int,
    fps: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    randomize_seed: bool,
    use_comfyui: bool,
    priority: str,
    progress=gr.Progress()
):
    """Main generation function with progress tracking."""
    import math

    add_log("=" * 50)
    add_log("🎬 Starting new generation job...")

    # Validate inputs
    if not prompt.strip():
        add_log("❌ Error: Prompt is required", "error")
        return None, None, get_logs_html(), "❌ Error: Prompt is required"

    if not script.strip():
        add_log("❌ Error: Script is required for audio", "error")
        return None, None, get_logs_html(), "❌ Error: Script is required"

    # Apply style preset
    if style_preset in STYLE_PRESETS:
        preset = STYLE_PRESETS[style_preset]
        prompt = prompt + preset["prompt_suffix"]
        cfg_scale = preset["cfg_scale"]
        checkpoint = preset["checkpoint"]
        add_log(f"🎨 Applied style preset: {style_preset}")

    # Generate seed
    if randomize_seed or seed == -1:
        seed = random.randint(0, 2**32 - 1)
        add_log(f"🎲 Generated random seed: {seed}")

    # Step 1: Generate audio
    progress(0.1, desc="Generating audio...")
    add_log("📝 Step 1/4: Generating audio...")
    audio_path = generate_audio(script, voice, rate, pitch)

    if not audio_path:
        return None, None, get_logs_html(), "❌ Audio generation failed"

    # Step 2: Calculate frame count
    progress(0.3, desc="Analyzing audio...")
    add_log("📊 Step 2/4: Calculating frame count...")

    try:
        audio_duration = get_audio_duration(audio_path)
        frame_count = math.ceil(audio_duration * fps)
        add_log(f"⏱️ Audio duration: {audio_duration:.2f}s | Frames: {frame_count} @ {fps} FPS", "success")
    except Exception as e:
        add_log(f"⚠️ Error calculating duration: {e}", "warning")
        frame_count = fps * 3
        audio_duration = 3.0

    # Step 3: Generate video
    progress(0.5, desc="Generating video frames...")
    add_log("🎥 Step 3/4: Video generation...")
    add_log(f"  📝 Prompt: {prompt[:60]}...")
    add_log(f"  📐 Resolution: {width}x{height}")
    add_log(f"  ⚙️ Steps: {steps} | CFG: {cfg_scale}")
    add_log(f"  🎨 Model: {checkpoint}")
    add_log(f"  🎬 Motion: {motion_module}")

    video_path = None

    if use_comfyui:
        # Try ComfyUI
        add_log("🔌 Attempting ComfyUI generation...")
        video_path = video_gen.generate_with_comfyui(
            prompt=prompt,
            negative_prompt=negative_prompt,
            audio_path=audio_path,
            width=width,
            height=height,
            fps=fps,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            checkpoint=checkpoint,
            motion_module=motion_module,
            frame_count=frame_count
        )

        if not video_path:
            add_log("⚠️ ComfyUI unavailable, using placeholder generator", "warning")

    if not video_path:
        # Use placeholder generator
        add_log("🎨 Generating placeholder video...")
        progress(0.7, desc="Creating animated video...")

        video_path = video_gen.generate_placeholder_video(
            prompt=prompt,
            audio_path=audio_path,
            width=width,
            height=height,
            fps=fps,
            duration=audio_duration
        )

    # Step 4: Finalize
    progress(0.9, desc="Finalizing...")
    add_log("✨ Step 4/4: Preparing output...")

    if video_path:
        # Save to history
        generation_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "video_path": video_path,
            "audio_path": audio_path,
            "settings": {
                "width": width,
                "height": height,
                "fps": fps,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed
            }
        })

        status_msg = f"""
✅ **Generation Complete!**

📊 **Summary:**
- 🎬 Video: {Path(video_path).name}
- 🎤 Audio: {Path(audio_path).name}
- ⏱️ Duration: {audio_duration:.2f}s
- 🎞️ Frames: {frame_count}
- 📐 Resolution: {width}x{height}
- 🎯 FPS: {fps}
- 🎲 Seed: {seed}

💡 **Tip:** Try different style presets for varied aesthetics!
        """

        add_log("🎉 Generation completed successfully!", "success")
        progress(1.0, desc="Done!")

        return video_path, audio_path, get_logs_html(), status_msg
    else:
        add_log("❌ Video generation failed", "error")
        return None, audio_path, get_logs_html(), "❌ Video generation failed"

# ============================================================================
# UI Helper Functions
# ============================================================================

def clear_inputs():
    """Clear all inputs."""
    return (
        "", NEGATIVE_PROMPT_DEFAULT, "", 0, 0,
        "Anime - Vibrant", CHECKPOINT_CHOICES[0], MOTION_MODULE_CHOICES[0],
        512, 512, 15, 20, 7.0, -1, True, False, "normal",
        get_logs_html()
    )

def apply_template(template_name: str):
    """Apply prompt template."""
    return PROMPT_TEMPLATES.get(template_name, "")

def apply_quality_preset(preset: str):
    """Apply quality preset."""
    presets = {
        "⚡ Draft (Fast)": (384, 384, 10, 15, 6.5),
        "📺 Standard (Balanced)": (512, 512, 15, 20, 7.0),
        "💎 High Quality (Slow)": (768, 768, 15, 30, 7.5),
        "🎬 Cinematic (4K Ready)": (1024, 576, 24, 35, 8.0),
    }
    return presets.get(preset, (512, 512, 15, 20, 7.0))

def get_generation_history():
    """Get generation history as HTML."""
    if not generation_history:
        return "<div style='text-align: center; color: #888; padding: 20px;'>No generations yet</div>"

    html = "<div style='display: grid; gap: 15px;'>"

    for item in reversed(generation_history[-10:]):  # Last 10
        timestamp = datetime.fromisoformat(item["timestamp"]).strftime("%I:%M %p")
        prompt = item["prompt"][:60] + "..." if len(item["prompt"]) > 60 else item["prompt"]

        html += f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; color: white;'>
            <div style='font-weight: bold; margin-bottom: 5px;'>🕐 {timestamp}</div>
            <div style='opacity: 0.9;'>"{prompt}"</div>
            <div style='margin-top: 8px; font-size: 11px; opacity: 0.8;'>
                {item['settings']['width']}×{item['settings']['height']} • 
                {item['settings']['fps']} FPS • 
                Seed: {item['settings']['seed']}
            </div>
        </div>
        """

    html += "</div>"
    return html

# ============================================================================
# Create Enhanced UI
# ============================================================================

def create_app():
    """Create the enhanced Gradio application."""


    with gr.Blocks(title="AI Anime Video Creator") as app:

        # Header
        gr.HTML("""
        <div class="main-header">🎬 AI Anime Video Creator</div>
        <div class="sub-header">
            ✨ Next-Generation Anime Video Synthesis Platform<br>
            🌍 Supports: Japanese • English • Hindi • Kannada • Korean • Chinese
        </div>
        """)

        with gr.Tabs() as tabs:

            # ========== TAB 1: CREATE ==========
            with gr.Tab("🎨 Create", id="create"):

                with gr.Row():
                    # LEFT PANEL
                    with gr.Column(scale=1):

                        # Visual Prompt
                        with gr.Group():
                            gr.Markdown("### 📝 Visual Description")

                            with gr.Row():
                                template_dropdown = gr.Dropdown(
                                    choices=list(PROMPT_TEMPLATES.keys()),
                                    label="Quick Templates",
                                    container=False
                                )
                                apply_template_btn = gr.Button("Apply", size="sm", scale=0)

                            prompt = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe your anime scene...\nExample: 1girl, anime, beautiful eyes, smiling, cherry blossoms",
                                lines=5,
                                max_lines=8
                            )

                            with gr.Accordion("⚙️ Advanced Prompt Settings", open=False):
                                negative_prompt = gr.Textbox(
                                    label="Negative Prompt",
                                    value=NEGATIVE_PROMPT_DEFAULT,
                                    lines=3
                                )

                                style_preset = gr.Dropdown(
                                    choices=list(STYLE_PRESETS.keys()),
                                    label="Style Preset",
                                    value="Anime - Vibrant"
                                )

                        # Audio Script
                        with gr.Group():
                            gr.Markdown("### 🎤 Audio & Voice")

                            script = gr.Textbox(
                                label="Script / Dialogue",
                                placeholder="Enter text for voice narration...\nSupports multiple languages!",
                                lines=4,
                                max_lines=6
                            )

                            voice = gr.Dropdown(
                                choices=VOICE_CHOICES,
                                label="Voice",
                                value="ja-JP-NanamiNeural",
                                filterable=True
                            )

                            with gr.Row():
                                rate = gr.Slider(-50, 50, 0, step=5, label="Speed (%)")
                                pitch = gr.Slider(-20, 20, 0, step=2, label="Pitch (Hz)")

                            with gr.Row():
                                preview_voice_btn = gr.Button("🔊 Preview Voice", size="sm")
                                audio_preview = gr.Audio(label="Preview", scale=2)

                        # Generation Settings
                        with gr.Group():
                            gr.Markdown("### ⚙️ Generation Settings")

                            quality_preset = gr.Radio(
                                ["⚡ Draft (Fast)", "📺 Standard (Balanced)", "💎 High Quality (Slow)", "🎬 Cinematic (4K Ready)"],
                                label="Quality Preset",
                                value="📺 Standard (Balanced)"
                            )

                            with gr.Accordion("🔧 Advanced Settings", open=False):
                                with gr.Row():
                                    width = gr.Slider(256, 1024, 512, step=64, label="Width")
                                    height = gr.Slider(256, 1024, 512, step=64, label="Height")

                                with gr.Row():
                                    fps = gr.Slider(8, 30, 15, step=1, label="FPS")
                                    steps = gr.Slider(10, 50, 20, step=5, label="Steps")

                                cfg_scale = gr.Slider(1, 20, 7.0, step=0.5, label="CFG Scale")

                                with gr.Row():
                                    seed = gr.Number(-1, label="Seed (-1 for random)")
                                    randomize_seed = gr.Checkbox(label="Randomize", value=True)

                                checkpoint = gr.Dropdown(
                                    choices=CHECKPOINT_CHOICES,
                                    label="Checkpoint Model",
                                    value=CHECKPOINT_CHOICES[0]
                                )

                                motion_module = gr.Dropdown(
                                    choices=MOTION_MODULE_CHOICES,
                                    label="Motion Module",
                                    value=MOTION_MODULE_CHOICES[0]
                                )

                                use_comfyui = gr.Checkbox(
                                    label="Use ComfyUI (if available)",
                                    value=False,
                                    info="Enable for production-quality videos"
                                )

                                priority = gr.Radio(
                                    ["high", "normal", "low"],
                                    label="Queue Priority",
                                    value="normal"
                                )

                        # Action Buttons
                        with gr.Row():
                            generate_btn = gr.Button(
                                "🎬 Generate Video",
                                variant="primary",
                                size="lg",
                                elem_classes=["generate-btn"]
                            )

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear All", size="sm")

                    # RIGHT PANEL
                    with gr.Column(scale=1):

                        gr.Markdown("### 🎥 Output")

                        # Video Output
                        video_output = gr.Video(
                            label="Generated Video",
                            height=450
                        )

                        # Audio Output
                        audio_output = gr.Audio(
                            label="Generated Audio"
                        )

                        # Status
                        status_output = gr.Markdown(
                            value="*Waiting for generation...*",
                            label="Status"
                        )

                        # Activity Log
                        with gr.Accordion("📋 Activity Log", open=True):
                            log_output = gr.HTML(
                                value=get_logs_html()
                            )
                            refresh_log_btn = gr.Button("🔄 Refresh Log", size="sm")

            # ========== TAB 2: HISTORY ==========
            with gr.Tab("📚 History", id="history"):

                gr.Markdown("## 📜 Generation History")

                history_output = gr.HTML(value=get_generation_history())
                refresh_history_btn = gr.Button("🔄 Refresh History")

            # ========== TAB 3: SETTINGS ==========
            with gr.Tab("⚙️ Settings", id="settings"):

                gr.Markdown("## ⚙️ Application Settings")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎨 UI Theme")
                        theme_selector = gr.Radio(
                            ["Light", "Dark", "Auto"],
                            value="Auto",
                            label="Theme Mode"
                        )

                        gr.Markdown("### 🔧 Default Settings")
                        default_fps = gr.Slider(8, 30, 15, label="Default FPS")
                        default_quality = gr.Dropdown(
                            ["Draft", "Standard", "High Quality"],
                            value="Standard",
                            label="Default Quality"
                        )

                    with gr.Column():
                        gr.Markdown("### 📊 Statistics")

                        stats_html = f"""
                        <div style="display: grid; gap: 15px;">
                            <div class="stat-card">
                                <div style="font-size: 2rem; font-weight: bold;">{len(generation_history)}</div>
                                <div>Total Generations</div>
                            </div>
                            <div class="stat-card">
                                <div style="font-size: 2rem; font-weight: bold;">{len(VOICE_CHOICES)}</div>
                                <div>Available Voices</div>
                            </div>
                            <div class="stat-card">
                                <div style="font-size: 2rem; font-weight: bold;">{len(PROMPT_TEMPLATES)}</div>
                                <div>Prompt Templates</div>
                            </div>
                        </div>
                        """
                        gr.HTML(stats_html)

            # ========== TAB 4: HELP ==========
            with gr.Tab("❓ Help", id="help"):

                gr.Markdown("""
                ## 📖 User Guide
                
                ### 🚀 Quick Start
                1. **Enter Visual Prompt**: Describe your anime scene
                2. **Add Voice Script**: Write dialogue for narration
                3. **Select Voice**: Choose from 13+ voices in 6 languages
                4. **Generate**: Click the generate button and wait
                
                ### 💡 Tips for Best Results
                - **Detailed Prompts**: Include specific details like "1girl, long pink hair, school uniform"
                - **Quality Presets**: Start with "Standard" for balanced speed/quality
                - **Voice Preview**: Test voices before generating full video
                - **Style Presets**: Experiment with different artistic styles
                
                ### 🎨 Supported Languages
                - 🇯🇵 Japanese (Best for anime)
                - 🇮🇳 Hindi (हिन्दी)
                - 🇮🇳 Kannada (ಕನ್ನಡ)
                - 🇺🇸 English
                - 🇰🇷 Korean
                - 🇨🇳 Chinese
                
                ### ⚙️ Advanced Features
                - **Seed Control**: Use the same seed to reproduce results
                - **CFG Scale**: Higher = more adherence to prompt (7-8 recommended)
                - **Motion Modules**: Control animation style
                - **ComfyUI Integration**: Enable for production quality (requires setup)
                
                ### 🐛 Troubleshooting
                - **No audio**: Check if script text is empty
                - **Slow generation**: Reduce resolution or use Draft preset
                - **Out of memory**: Lower resolution or FPS
                
                ### 📞 Support
                For issues or questions, check the logs in the Activity Log panel.
                """)

        # ========================================
        # EVENT BINDINGS
        # ========================================

        # Generate
        generate_btn.click(
            fn=on_generate,
            inputs=[
                prompt, negative_prompt, script, voice,
                rate, pitch, style_preset, checkpoint, motion_module,
                width, height, fps, steps, cfg_scale,
                seed, randomize_seed, use_comfyui, priority
            ],
            outputs=[video_output, audio_output, log_output, status_output]
        )

        # Preview voice
        preview_voice_btn.click(
            fn=preview_voice,
            inputs=[script, voice, rate, pitch],
            outputs=[audio_preview, log_output]
        )

        # Apply template
        apply_template_btn.click(
            fn=apply_template,
            inputs=[template_dropdown],
            outputs=[prompt]
        )

        # Quality preset
        quality_preset.change(
            fn=apply_quality_preset,
            inputs=[quality_preset],
            outputs=[width, height, fps, steps, cfg_scale]
        )

        # Clear
        clear_btn.click(
            fn=clear_inputs,
            outputs=[
                prompt, negative_prompt, script,
                rate, pitch, style_preset, checkpoint, motion_module,
                width, height, fps, steps, cfg_scale,
                seed, randomize_seed, use_comfyui, priority,
                log_output
            ]
        )

        # Refresh log
        refresh_log_btn.click(
            fn=get_logs_html,
            outputs=[log_output]
        )

        # Refresh history
        refresh_history_btn.click(
            fn=get_generation_history,
            outputs=[history_output]
        )

    return app


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    import io

    # Fix Unicode for Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "="*60)
    print("  🎬 AI ANIME VIDEO CREATOR")
    print("  Next-Generation Video Synthesis Platform")
    print("="*60)
    print("\n🌍 Supported Languages:")
    print("  • Japanese (日本語) • English • Hindi (हिन्दी)")
    print("  • Kannada (ಕನ್ನಡ) • Korean (한국어) • Chinese (中文)")
    print("\n🚀 Starting application...\n")

    app = create_app()

    # Enable queue for concurrent requests
    app.queue(
        max_size=20,
        default_concurrency_limit=3
    )

    # Launch with auto-port selection
    launched = False
    for port in range(7860, 7871):
        try:
            print(f"🌐 Attempting to start on port {port}...")
            app.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=True,  # Enable share functionality
                show_error=True,
                inbrowser=True
            )
            launched = True
            break
        except OSError:
            print(f"   Port {port} busy, trying next...")
            continue

    if not launched:
        print("❌ Could not find available port. Please close other applications.")

