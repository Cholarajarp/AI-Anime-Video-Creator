"""
AI Video Creator - Main Application Entry Point
================================================
This is the main Gradio application that combines all UI components
and connects them to the backend services.
"""

import gradio as gr
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# ============================================================================
# Simplified Standalone App (Works without ComfyUI)
# ============================================================================

# Default voices with Hindi and Kannada support
VOICE_CHOICES = [
    ("Nanami (Japanese Female) - Energetic Anime Girl", "ja-JP-NanamiNeural"),
    ("Aoi (Japanese Female) - Gentle & Soft", "ja-JP-AoiNeural"),
    ("Keita (Japanese Male) - Young & Dynamic", "ja-JP-KeitaNeural"),
    ("Daichi (Japanese Male) - Deep & Confident", "ja-JP-DaichiNeural"),
    ("Aria (English Female) - Narrator Style", "en-US-AriaNeural"),
    ("Jenny (English Female) - Friendly & Warm", "en-US-JennyNeural"),
    ("Guy (English Male) - Professional", "en-US-GuyNeural"),
    ("SunHi (Korean Female) - K-Drama Style", "ko-KR-SunHiNeural"),
    ("Xiaoxiao (Chinese Female) - Warm & Expressive", "zh-CN-XiaoxiaoNeural"),
    ("Swara (Hindi Female) - Expressive & Warm", "hi-IN-SwaraNeural"),
    ("Madhur (Hindi Male) - Professional & Clear", "hi-IN-MadhurNeural"),
    ("Sapna (Kannada Female) - Melodic & Pleasant", "kn-IN-SapnaNeural"),
    ("Gagan (Kannada Male) - Deep & Authoritative", "kn-IN-GaganNeural"),
]

PROMPT_TEMPLATES = {
    "Anime Girl - Portrait": "1girl, solo, anime, looking at viewer, smile, beautiful face, detailed eyes, school uniform, cherry blossoms, soft lighting",
    "Anime Boy - Dynamic": "1boy, solo, anime, dynamic pose, cool expression, spiky hair, jacket, urban background, dramatic lighting",
    "Cute Chibi": "chibi, 1girl, kawaii, big eyes, pastel colors, simple background, happy expression, bouncing",
    "Fantasy Scene": "anime, fantasy, magical forest, glowing particles, mystical atmosphere, ethereal lighting, detailed background",
    "Action Pose": "anime, dynamic action pose, motion blur, intense expression, energy effects, dramatic angle",
}

NEGATIVE_PROMPT_DEFAULT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# Model choices
CHECKPOINT_CHOICES = [
    "dreamshaper_8.safetensors",
    "meinamix_v11.safetensors",
    "counterfeit_v3.safetensors",
    "anything_v5.safetensors"
]

MOTION_MODULE_CHOICES = [
    "mm_sd_v15_v2.ckpt",
    "mm_sd_v15_v3.ckpt"
]

# Global state
generation_logs = []


def add_log(message: str, level: str = "info"):
    """Add a log entry."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    icon = icons.get(level, "•")
    generation_logs.append(f"[{timestamp}] {icon} {message}")
    if len(generation_logs) > 100:
        generation_logs.pop(0)
    logger.info(message)


def get_logs_html():
    """Get formatted logs HTML."""
    if not generation_logs:
        return "<div style='color: #888; text-align: center; padding: 20px;'>No activity yet...</div>"

    log_html = "<div style='font-family: monospace; font-size: 12px;'>"
    for log in reversed(generation_logs[-50:]):
        log_html += f"<div style='padding: 4px 0; border-bottom: 1px solid #333;'>{log}</div>"
    log_html += "</div>"
    return log_html


async def generate_audio_async(text: str, voice_id: str, output_path: Path):
    """Generate audio using Edge-TTS."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text=text, voice=voice_id)
        await communicate.save(str(output_path))
        return True
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        return False


def generate_audio(text: str, voice_id: str, rate: int = 0, pitch: int = 0):
    """Synchronous wrapper for audio generation."""
    import uuid
    from pathlib import Path

    # Validate input
    if not text or not text.strip():
        add_log("Audio error: Empty text provided", "error")
        return None

    # Create temp directory
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)

    output_path = temp_dir / f"audio_{uuid.uuid4().hex[:8]}.mp3"

    try:
        import edge_tts

        # Format rate and pitch
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

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate())
        loop.close()

        if output_path.exists() and output_path.stat().st_size > 0:
            add_log(f"Audio generated: {output_path.name} ({output_path.stat().st_size / 1024:.1f} KB)", "success")
            return str(output_path)
        else:
            add_log("Audio generation failed: Empty file", "error")
            return None

    except ImportError:
        add_log("edge-tts not installed. Run: pip install edge-tts", "error")
        return None
    except Exception as e:
        add_log(f"Audio error: {str(e)}", "error")
        import traceback
        logger.error(traceback.format_exc())
        return None


def preview_voice(text: str, voice_id: str, rate: int, pitch: int):
    """Generate voice preview."""
    if not text.strip():
        return None, get_logs_html()

    preview_text = text[:200] if len(text) > 200 else text
    add_log(f"Generating voice preview with {voice_id}...")

    audio_path = generate_audio(preview_text, voice_id, rate, pitch)
    return audio_path, get_logs_html()


def on_generate(
    prompt: str,
    negative_prompt: str,
    script: str,
    voice: str,
    rate: int,
    pitch: int,
    checkpoint: str,
    motion_module: str,
    width: int,
    height: int,
    fps: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    randomize_seed: bool,
    priority: str
):
    """Handle generate button click."""
    import math

    add_log("=" * 40)
    add_log("Starting new generation job...")

    # Validate inputs
    if not prompt.strip():
        add_log("Error: Prompt is required", "error")
        return None, None, get_logs_html(), "Error: Prompt is required"

    if not script.strip():
        add_log("Error: Script is required for audio", "error")
        return None, None, get_logs_html(), "Error: Script is required"

    # Generate random seed if needed
    if randomize_seed or seed == -1:
        seed = random.randint(0, 2**32 - 1)
        add_log(f"Generated random seed: {seed}")

    # Step 1: Generate audio
    add_log("Step 1/4: Generating audio...")
    audio_path = generate_audio(script, voice, rate, pitch)

    if not audio_path:
        return None, None, get_logs_html(), "Audio generation failed"

    # Step 2: Calculate frame count from audio duration
    add_log("Step 2/4: Calculating frame count...")
    try:
        # Try multiple methods to get audio duration
        audio_duration = None

        # Method 1: Try pydub (requires ffmpeg/ffprobe)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(audio_path)
            audio_duration = len(audio) / 1000.0
            add_log(f"Audio duration (pydub): {audio_duration:.2f}s")
        except Exception as e:
            logger.debug(f"pydub failed: {e}")

        # Method 2: Try mutagen (pure Python, no ffmpeg needed)
        if audio_duration is None:
            try:
                from mutagen.mp3 import MP3
                audio = MP3(audio_path)
                audio_duration = audio.info.length
                add_log(f"Audio duration (mutagen): {audio_duration:.2f}s")
            except ImportError:
                add_log("Installing mutagen for audio analysis...", "info")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "mutagen"])
                from mutagen.mp3 import MP3
                audio = MP3(audio_path)
                audio_duration = audio.info.length
                add_log(f"Audio duration (mutagen): {audio_duration:.2f}s")
            except Exception as e:
                logger.debug(f"mutagen failed: {e}")

        # Method 3: Estimate from file size (rough approximation)
        if audio_duration is None:
            import os
            file_size = os.path.getsize(audio_path)
            # Rough estimate: ~16KB per second for typical TTS at 64kbps
            audio_duration = file_size / (16 * 1024)
            add_log(f"Audio duration (estimated): {audio_duration:.2f}s", "warning")

        frame_count = math.ceil(audio_duration * fps)
        add_log(f"Frames needed: {frame_count} ({fps} FPS)")

    except Exception as e:
        add_log(f"Error calculating duration: {e}", "error")
        frame_count = fps * 3  # Default to 3 seconds
        audio_duration = 3.0
        add_log(f"Using default duration: {audio_duration}s", "warning")

    # Step 3: Video generation (would use ComfyUI in production)
    add_log("Step 3/4: Video generation...")
    add_log(f"  Prompt: {prompt[:50]}...", "info")
    add_log(f"  Resolution: {width}x{height}", "info")
    add_log(f"  Steps: {steps}, CFG: {cfg_scale}", "info")
    add_log(f"  Model: {checkpoint}", "info")
    add_log(f"  Motion: {motion_module}", "info")

    # Check if ComfyUI is available
    add_log("Note: ComfyUI connection required for video generation", "warning")
    add_log("Please ensure ComfyUI is running on port 8188", "warning")

    # Step 4: Return results
    add_log("Step 4/4: Preparing output...")

    status_msg = f"""
    ✅ Audio Generated Successfully!
    
    📊 Generation Summary:
    - Audio Duration: {audio_duration:.2f}s
    - Required Frames: {frame_count}
    - Resolution: {width}x{height}
    - FPS: {fps}
    - Seed: {seed}
    
    ⚠️ Video Generation Status:
    ComfyUI integration pending. 
    Please start ComfyUI with AnimateDiff to generate video.
    
    Command: python main.py --listen 0.0.0.0 --port 8188
    """

    add_log("Generation job completed!", "success")

    return None, audio_path, get_logs_html(), status_msg


def clear_inputs():
    """Clear all inputs."""
    return (
        "",  # prompt
        NEGATIVE_PROMPT_DEFAULT,  # negative
        "",  # script
        0,   # rate
        0,   # pitch
        512, # width
        512, # height
        15,  # fps
        20,  # steps
        7.0, # cfg
        -1,  # seed
        True, # randomize
        "normal",  # priority
        get_logs_html()  # logs
    )


def apply_template(template_name: str):
    """Apply a prompt template."""
    return PROMPT_TEMPLATES.get(template_name, "")


def apply_quality_preset(preset: str):
    """Apply quality preset."""
    presets = {
        "Draft": (384, 384, 15, 6.0),
        "Standard": (512, 512, 20, 7.0),
        "High Quality": (768, 768, 30, 7.5)
    }
    return presets.get(preset, (512, 512, 20, 7.0))


# ============================================================================
# Build UI
# ============================================================================

def create_app():
    """Create the main Gradio application."""

    # Custom CSS
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 20px;
    }
    .generate-btn {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        font-size: 1.2rem !important;
        padding: 12px 24px !important;
    }
    .status-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        font-family: monospace;
    }
    """

    # Store theme and css for launch
    app_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple"
    )

    with gr.Blocks(
        title="Anime Video Creator"
    ) as app:

        # Header
        gr.HTML("""
        <div class="main-header">🎬 AI Anime Video Creator</div>
        <div class="sub-header">Next-Generation Anime Video Synthesis Platform • Supports Hindi & Kannada</div>
        """)

        with gr.Row():
            # Left Column - Controls
            with gr.Column(scale=1):
                gr.Markdown("## 🎨 Generation Controls")

                # Prompt Section
                with gr.Accordion("📝 Visual Prompt", open=True):
                    with gr.Row():
                        template_dropdown = gr.Dropdown(
                            choices=list(PROMPT_TEMPLATES.keys()),
                            label="Quick Templates",
                            value=None
                        )
                        apply_btn = gr.Button("Apply", size="sm")

                    prompt = gr.Textbox(
                        label="Visual Prompt",
                        placeholder="Describe your anime video...\nExample: 1girl, anime, smile, waving, pink hair",
                        lines=4
                    )

                    with gr.Accordion("Negative Prompt", open=False):
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value=NEGATIVE_PROMPT_DEFAULT,
                            lines=2
                        )

                # Audio Script Section
                with gr.Accordion("🎤 Audio Script", open=True):
                    script = gr.Textbox(
                        label="Script / Dialogue",
                        placeholder="Enter text to convert to speech...\nSupports: Japanese, English, Hindi, Kannada, Korean, Chinese",
                        lines=4
                    )

                    voice = gr.Dropdown(
                        choices=VOICE_CHOICES,
                        label="Voice",
                        value="ja-JP-NanamiNeural"
                    )

                    with gr.Row():
                        rate = gr.Slider(-50, 50, 0, step=5, label="Speech Rate (%)")
                        pitch = gr.Slider(-20, 20, 0, step=2, label="Pitch (Hz)")

                    preview_btn = gr.Button("🔊 Preview Voice", size="sm")
                    audio_preview = gr.Audio(label="Voice Preview", visible=True)

                # Model Settings
                with gr.Accordion("🤖 Model Settings", open=False):
                    checkpoint = gr.Dropdown(
                        choices=CHECKPOINT_CHOICES,
                        label="Checkpoint",
                        value=CHECKPOINT_CHOICES[0]
                    )
                    motion_module = gr.Dropdown(
                        choices=MOTION_MODULE_CHOICES,
                        label="Motion Module",
                        value=MOTION_MODULE_CHOICES[0]
                    )

                # Generation Settings
                with gr.Accordion("⚙️ Generation Settings", open=False):
                    quality_preset = gr.Radio(
                        ["Draft", "Standard", "High Quality"],
                        label="Quality Preset",
                        value="Standard"
                    )

                    with gr.Row():
                        width = gr.Slider(256, 1024, 512, step=64, label="Width")
                        height = gr.Slider(256, 1024, 512, step=64, label="Height")

                    with gr.Row():
                        fps = gr.Slider(8, 30, 15, step=1, label="FPS")
                        steps = gr.Slider(10, 50, 20, step=5, label="Steps")

                    with gr.Row():
                        cfg_scale = gr.Slider(1, 20, 7.0, step=0.5, label="CFG Scale")
                        seed = gr.Number(-1, label="Seed (-1=random)")

                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

                    priority = gr.Radio(
                        ["high", "normal", "low"],
                        label="Priority",
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
                    clear_btn = gr.Button("🗑️ Clear", size="lg")

            # Right Column - Output
            with gr.Column(scale=1):
                gr.Markdown("## 🖥️ Output")

                # Video Output
                video_output = gr.Video(
                    label="Generated Video",
                    height=400
                )

                # Audio Output
                audio_output = gr.Audio(
                    label="Generated Audio"
                )

                # Status
                status_output = gr.Textbox(
                    label="Status",
                    lines=10,
                    interactive=False
                )

                # Activity Log
                with gr.Accordion("📝 Activity Log", open=True):
                    log_output = gr.HTML(
                        value="<div style='color: #888; text-align: center; padding: 20px;'>Waiting for activity...</div>"
                    )

        # ========================================
        # Event Bindings
        # ========================================

        # Generate button
        generate_btn.click(
            fn=on_generate,
            inputs=[
                prompt, negative_prompt, script, voice,
                rate, pitch, checkpoint, motion_module,
                width, height, fps, steps, cfg_scale,
                seed, randomize_seed, priority
            ],
            outputs=[video_output, audio_output, log_output, status_output]
        )

        # Preview voice
        preview_btn.click(
            fn=preview_voice,
            inputs=[script, voice, rate, pitch],
            outputs=[audio_preview, log_output]
        )

        # Apply template
        apply_btn.click(
            fn=apply_template,
            inputs=[template_dropdown],
            outputs=[prompt]
        )

        # Quality preset
        quality_preset.change(
            fn=apply_quality_preset,
            inputs=[quality_preset],
            outputs=[width, height, steps, cfg_scale]
        )

        # Clear button
        clear_btn.click(
            fn=clear_inputs,
            outputs=[
                prompt, negative_prompt, script,
                rate, pitch, width, height, fps, steps,
                cfg_scale, seed, randomize_seed, priority,
                log_output
            ]
        )

    return app


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    import io

    # Fix Unicode output on Windows
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "="*50)
    print("  AI Anime Video Creator")
    print("  Starting application...")
    print("="*50 + "\n")

    app = create_app()

    print("Server starting at http://localhost:7860")
    print("Supports: Japanese, English, Hindi, Kannada, Korean, Chinese\n")

    app.queue(max_size=20)

    # Try port 7860, fallback to 7861-7870 if busy
    for port in range(7860, 7871):
        try:
            app.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True
            )
            break
        except OSError:
            print(f"Port {port} is busy, trying next...")
            continue

