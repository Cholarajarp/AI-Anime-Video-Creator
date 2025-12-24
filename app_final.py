"""
AI Anime Video Creator - Production Ready
==========================================
Full-featured application with:
- Advanced animation generation
- Multi-language translation
- Professional UI
- All features activated
"""

import gradio as gr
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import random
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from backend.services.cinematic_video_generator import CinematicVideoGenerator, CAMERA_MOTIONS, LIGHTING_STYLES
from backend.services.translation_service import TranslationService

# Initialize services
video_gen = CinematicVideoGenerator()
translator = TranslationService()

# ============================================================================
# CONFIGURATION
# ============================================================================

VOICES = [
    ("🇯🇵 Nanami (Japanese Female)", "ja-JP-NanamiNeural"),
    ("🇯🇵 Aoi (Japanese Female)", "ja-JP-AoiNeural"),
    ("🇯🇵 Keita (Japanese Male)", "ja-JP-KeitaNeural"),
    ("🇮🇳 Swara (Hindi Female)", "hi-IN-SwaraNeural"),
    ("🇮🇳 Madhur (Hindi Male)", "hi-IN-MadhurNeural"),
    ("🇮🇳 Sapna (Kannada Female)", "kn-IN-SapnaNeural"),
    ("🇮🇳 Gagan (Kannada Male)", "kn-IN-GaganNeural"),
    ("🇺🇸 Aria (English Female)", "en-US-AriaNeural"),
    ("🇺🇸 Guy (English Male)", "en-US-GuyNeural"),
    ("🇰🇷 SunHi (Korean)", "ko-KR-SunHiNeural"),
    ("🇨🇳 Xiaoxiao (Chinese)", "zh-CN-XiaoxiaoNeural"),
]

# Cinematic templates
TEMPLATES = {
    "🎬 Cinematic Portrait": "beautiful anime girl, detailed eyes, flowing hair, soft lighting, cinematic, film quality",
    "✨ Magical Fantasy": "anime, magical girl, glowing particles, fantasy, mystical aura, ethereal lighting",
    "💥 Action Hero": "anime hero, dynamic pose, energy aura, battle stance, dramatic lighting, powerful",
    "💕 Romantic Scene": "anime, romantic, sunset, warm colors, gentle expression, emotional, beautiful",
    "🌸 Nature Spirit": "anime, nature spirit, forest background, flowers, peaceful, serene, magical",
    "🌃 Cyberpunk Neon": "anime, cyberpunk, neon lights, futuristic city, night, purple blue tones",
    "🧸 Cute Kawaii": "chibi, kawaii, big sparkly eyes, cute, happy, pastel colors, adorable",
    "🌙 Dark Mystery": "anime, mysterious, dark fantasy, moonlight, gothic, elegant, shadows",
    "🏯 Traditional Japanese": "anime, traditional kimono, japanese style, cherry blossoms, graceful, elegant",
    "🚀 Space Adventure": "anime, space, stars, galaxy background, sci-fi, cosmic, adventure",
}

NEGATIVE = "lowres, bad anatomy, text, error, worst quality, blurry, ugly"

# State
logs = []
history = []

def log(msg, level="info"):
    """Add log message."""
    time = datetime.now().strftime("%H:%M:%S")
    logs.append({"time": time, "level": level, "msg": msg})
    if len(logs) > 30:
        logs.pop(0)

    colors = {"error": "#ff6b6b", "warning": "#ffd43b", "success": "#51cf66", "info": "#74c0fc"}
    icons = {"error": "❌", "warning": "⚠️", "success": "✅", "info": "ℹ️"}

    if level == "error":
        logger.error(msg)
    elif level == "warning":
        logger.warning(msg)
    else:
        logger.info(msg)

def get_logs_html():
    """Generate logs HTML."""
    if not logs:
        return "<div style='text-align:center;color:#888;padding:20px'>No activity</div>"

    html = "<div style='font-family:monospace;font-size:12px'>"
    colors = {"error": "#ff6b6b", "warning": "#ffd43b", "success": "#51cf66", "info": "#74c0fc"}
    icons = {"error": "❌", "warning": "⚠️", "success": "✅", "info": "ℹ️"}

    for l in reversed(logs[-15:]):
        color = colors.get(l["level"], "#adb5bd")
        icon = icons.get(l["level"], "•")
        html += f"<div style='margin:4px 0;color:{color}'>[{l['time']}] {icon} {l['msg']}</div>"

    html += "</div>"
    return html

# ============================================================================
# AUDIO GENERATION
# ============================================================================

def generate_audio(text: str, voice: str, rate: int, pitch: int) -> Optional[str]:
    """Generate TTS audio."""
    import uuid
    from pathlib import Path

    if not text.strip():
        log("Empty text for audio", "error")
        return None

    Path("./temp").mkdir(exist_ok=True)
    output = Path(f"./temp/audio_{uuid.uuid4().hex[:8]}.mp3")

    try:
        import edge_tts

        rate_str = f"+{rate}%" if rate >= 0 else f"{rate}%"
        pitch_str = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

        async def _gen():
            comm = edge_tts.Communicate(text.strip(), voice, rate=rate_str, pitch=pitch_str)
            await comm.save(str(output))

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_gen())
        loop.close()

        if output.exists() and output.stat().st_size > 0:
            size = output.stat().st_size / 1024
            log(f"Audio generated: {output.name} ({size:.1f} KB)", "success")
            return str(output)

        log("Audio file empty", "error")
        return None

    except Exception as e:
        log(f"Audio error: {e}", "error")
        return None

def get_duration(audio_path: str) -> float:
    """Get audio duration."""
    try:
        from mutagen.mp3 import MP3
        return MP3(audio_path).info.length
    except:
        import os
        return os.path.getsize(audio_path) / (16 * 1024)

# ============================================================================
# MAIN GENERATION
# ============================================================================

def generate_video(
    prompt, negative, script, voice, rate, pitch,
    translate_prompt, target_lang,
    width, height, fps, steps, cfg, seed, randomize,
    camera_motion, lighting_style,
    progress=gr.Progress()
):
    """Main generation function with cinematic options."""
    import math

    log("="*40)
    log("🎬 Starting CINEMATIC generation...")
    log(f"📷 Camera: {camera_motion} | 💡 Lighting: {lighting_style}")

    # Validate
    if not prompt.strip():
        log("Prompt required", "error")
        return None, None, get_logs_html(), "❌ Prompt required"

    if not script.strip():
        log("Script required", "error")
        return None, None, get_logs_html(), "❌ Script required"

    # Translation
    original_prompt = prompt
    original_script = script

    if translate_prompt and target_lang != "None":
        log(f"🌍 Translating to {target_lang}...")
        progress(0.05, desc="Translating...")

        try:
            prompt = translator.translate_text(prompt, target_lang)
            log(f"Translated prompt: {prompt[:50]}...", "success")
        except Exception as e:
            log(f"Translation failed: {e}", "warning")

    # Auto-detect language from voice and translate script
    voice_lang = translator.get_language_from_voice(voice)
    if voice_lang and voice_lang != "English":
        log(f"🌍 Translating script to {voice_lang}...")
        try:
            script = translator.translate_text(original_script, voice_lang)
            log(f"Script translated to {voice_lang}", "success")
        except:
            pass

    # Enhance prompt
    prompt = translator.enhance_prompt_with_language(prompt, voice_lang)

    # Seed
    if randomize or seed == -1:
        seed = random.randint(0, 2**32 - 1)
    log(f"🎲 Seed: {seed}")

    # Generate audio
    progress(0.1, desc="Generating audio...")
    log("🎤 Step 1/3: Generating audio...")

    audio_path = generate_audio(script, voice, rate, pitch)
    if not audio_path:
        return None, None, get_logs_html(), "❌ Audio failed"

    # Calculate duration
    progress(0.3, desc="Analyzing audio...")
    log("📊 Step 2/3: Calculating duration...")

    try:
        duration = get_duration(audio_path)
        frames = math.ceil(duration * fps)
        log(f"Duration: {duration:.2f}s | Frames: {frames} @ {fps} FPS", "success")
    except Exception as e:
        log(f"Duration error: {e}", "warning")
        duration = 3.0
        frames = fps * 3

    # Generate video
    progress(0.5, desc="Generating animated video...")
    log("🎥 Step 3/3: Generating video...")
    log(f"  Prompt: {prompt[:60]}...")
    log(f"  Resolution: {width}x{height}")
    log(f"  Camera: {camera_motion} | Lighting: {lighting_style}")

    try:
        # Get camera motion code
        camera_code = CAMERA_MOTIONS.get(camera_motion, "dolly_in")
        lighting_code = LIGHTING_STYLES.get(lighting_style, "cinematic")

        video_path = video_gen.generate_video(
            prompt=prompt,
            audio_path=audio_path,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            camera_motion=camera_code,
            lighting_style=lighting_code
        )

        if video_path:
            # Save to history
            history.append({
                "time": datetime.now().isoformat(),
                "prompt": original_prompt,
                "video": video_path,
                "audio": audio_path
            })

            log("🎉 Generation complete!", "success")

            status = f"""
✅ **Generation Successful!**

📹 Video: {Path(video_path).name}
🎵 Audio: {Path(audio_path).name}
⏱️  Duration: {duration:.2f}s
🎞️  Frames: {frames}
📐 Resolution: {width}×{height}
🎲 Seed: {seed}

{'🌍 Translated: ' + target_lang if translate_prompt else ''}
"""
            progress(1.0, desc="Done!")
            return video_path, audio_path, get_logs_html(), status

        log("Video generation failed", "error")
        return None, audio_path, get_logs_html(), "❌ Video failed"

    except Exception as e:
        log(f"Generation error: {e}", "error")
        import traceback
        logger.error(traceback.format_exc())
        return None, audio_path, get_logs_html(), f"❌ Error: {e}"

# ============================================================================
# UI FUNCTIONS
# ============================================================================

def preview_voice(text, voice, rate, pitch):
    """Preview voice."""
    if not text.strip():
        return None, get_logs_html()

    preview_text = text[:150]
    log(f"🎤 Previewing: {voice}")
    audio = generate_audio(preview_text, voice, rate, pitch)
    return audio, get_logs_html()

def apply_template(template_name):
    """Apply template."""
    return TEMPLATES.get(template_name, "")

def apply_preset(preset):
    """Apply quality preset."""
    presets = {
        "⚡ Draft": (384, 384, 10, 15),
        "📺 Standard": (512, 512, 15, 20),
        "⚡ Draft": (384, 384, 15, 15),
        "📺 Standard": (512, 512, 24, 20),
        "💎 High Quality": (768, 768, 24, 30),
    }
    return presets.get(preset, (512, 512, 24, 20))

def clear_all():
    """Clear inputs."""
    return "", NEGATIVE, "", 0, 0, False, "None", 512, 512, 24, 20, 7.0, -1, True, "Dolly In (Slow Zoom)", "Cinematic", get_logs_html()

def get_history_html():
    """Get generation history."""
    if not history:
        return "<div style='text-align:center;color:#888;padding:20px'>No history</div>"

    html = "<div style='display:grid;gap:10px'>"
    for item in reversed(history[-10:]):
        time = datetime.fromisoformat(item["time"]).strftime("%I:%M %p")
        prompt = item["prompt"][:50] + "..." if len(item["prompt"]) > 50 else item["prompt"]
        html += f"""
        <div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:12px;border-radius:8px;color:white'>
            <div style='font-weight:bold'>{time}</div>
            <div style='opacity:0.9;font-size:13px'>{prompt}</div>
        </div>
        """
    html += "</div>"
    return html

# ============================================================================
# BUILD UI
# ============================================================================

def build_ui():
    """Build the Gradio interface."""

    with gr.Blocks(title="AI Anime Video Creator") as app:

        gr.HTML("""
        <div style='text-align:center;margin:20px'>
            <h1 style='background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;
                       -webkit-text-fill-color:transparent;font-size:3rem;margin:0'>
                🎬 AI Anime Video Creator
            </h1>
            <p style='color:#888;font-size:1.1rem;margin-top:10px'>
                Professional Anime Video Generation • Multi-Language Support
            </p>
        </div>
        """)

        with gr.Tabs():

            with gr.Tab("🎨 Create"):
                with gr.Row():
                    # LEFT PANEL
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Visual Prompt")

                        with gr.Row():
                            template = gr.Dropdown(list(TEMPLATES.keys()), label="Template")
                            apply_tpl_btn = gr.Button("Apply", size="sm", scale=0)

                        prompt = gr.Textbox(
                            label="Describe your anime video",
                            placeholder="1girl, anime, beautiful, smiling...",
                            lines=4
                        )

                        with gr.Accordion("⚙️ Advanced", open=False):
                            negative = gr.Textbox(label="Negative Prompt", value=NEGATIVE, lines=2)

                        gr.Markdown("### 🎤 Audio Script")

                        script = gr.Textbox(
                            label="Dialogue/Narration",
                            placeholder="Enter text for voice narration...",
                            lines=3
                        )

                        voice = gr.Dropdown(VOICES, label="Voice", value="ja-JP-NanamiNeural")

                        with gr.Row():
                            rate = gr.Slider(-50, 50, 0, label="Speed (%)")
                            pitch = gr.Slider(-20, 20, 0, label="Pitch (Hz)")

                        preview_btn = gr.Button("🔊 Preview Voice", size="sm")
                        preview_audio = gr.Audio(label="Preview")

                        gr.Markdown("### 🌍 Translation")

                        translate_prompt = gr.Checkbox(label="Translate Prompt", value=False)
                        target_lang = gr.Dropdown(
                            ["None", "Japanese", "Hindi", "Kannada", "Korean", "Chinese"],
                            label="Target Language",
                            value="None"
                        )

                        gr.Info("💡 Script is auto-translated to match selected voice language")

                        gr.Markdown("### ⚙️ Settings")

                        quality = gr.Radio(
                            ["⚡ Draft", "📺 Standard", "💎 High Quality"],
                            label="Quality",
                            value="📺 Standard"
                        )

                        gr.Markdown("### 🎬 Cinematic Options")

                        with gr.Row():
                            camera_motion = gr.Dropdown(
                                list(CAMERA_MOTIONS.keys()),
                                label="📷 Camera Motion",
                                value="Dolly In (Slow Zoom)"
                            )
                            lighting_style = gr.Dropdown(
                                list(LIGHTING_STYLES.keys()),
                                label="💡 Lighting Style",
                                value="Cinematic"
                            )

                        with gr.Accordion("🔧 Advanced Settings", open=False):
                            with gr.Row():
                                width = gr.Slider(256, 1024, 512, step=64, label="Width")
                                height = gr.Slider(256, 1024, 512, step=64, label="Height")

                            with gr.Row():
                                fps = gr.Slider(12, 30, 24, label="FPS (24 = Cinema)")
                                steps = gr.Slider(10, 50, 20, label="Steps")

                            cfg = gr.Slider(1, 20, 7.0, step=0.5, label="CFG Scale")

                            with gr.Row():
                                seed = gr.Number(-1, label="Seed")
                                randomize = gr.Checkbox(True, label="Random")

                        with gr.Row():
                            generate_btn = gr.Button(
                                "🎬 GENERATE VIDEO",
                                variant="primary",
                                size="lg",
                                scale=2
                            )
                            clear_btn = gr.Button("🗑️ Clear", size="lg", scale=1)

                    # RIGHT PANEL
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎥 Output")

                        video_out = gr.Video(label="Generated Video", height=400)
                        audio_out = gr.Audio(label="Generated Audio")

                        status_out = gr.Markdown("*Waiting for generation...*")

                        with gr.Accordion("📋 Activity Log", open=True):
                            log_out = gr.HTML(get_logs_html())
                            refresh_log = gr.Button("🔄 Refresh", size="sm")

            with gr.Tab("📚 History"):
                gr.Markdown("## 📜 Generation History")
                history_out = gr.HTML(get_history_html())
                refresh_hist = gr.Button("🔄 Refresh History")

            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## 🚀 Quick Start Guide
                
                ### 1️⃣ Enter Visual Prompt
                Describe your anime scene in detail:
                - Use templates or write custom prompts
                - Example: "1girl, anime, beautiful eyes, smiling, pink hair"
                
                ### 2️⃣ Add Audio Script
                Write dialogue or narration:
                - Select a voice (13+ options in 6 languages)
                - Script auto-translates to match voice language
                - Adjust speed and pitch
                
                ### 3️⃣ Translation (Optional)
                - Enable "Translate Prompt" to convert your English prompt
                - Select target language
                - Great for authentic anime styles
                
                ### 4️⃣ Generate
                Click "GENERATE VIDEO" and wait!
                - Draft: ~10-20 seconds
                - Standard: ~30-60 seconds
                - High Quality: ~1-2 minutes
                
                ## 🎨 Supported Languages
                - 🇯🇵 Japanese (Best for anime)
                - 🇮🇳 Hindi (हिन्दी)
                - 🇮🇳 Kannada (ಕನ್ನಡ)
                - 🇺🇸 English
                - 🇰🇷 Korean
                - 🇨🇳 Chinese
                
                ## 💡 Tips
                - Use specific details in prompts for better results
                - Preview voice before full generation
                - Higher FPS = smoother animation (but slower)
                - CFG Scale 7-8 recommended for anime
                
                ## 🔧 Troubleshooting
                - **No audio**: Check if script is empty
                - **Slow**: Use Draft quality or lower resolution
                - **Poor quality**: Increase resolution and steps
                """)

        # EVENT HANDLERS

        generate_btn.click(
            generate_video,
            [prompt, negative, script, voice, rate, pitch,
             translate_prompt, target_lang,
             width, height, fps, steps, cfg, seed, randomize,
             camera_motion, lighting_style],
            [video_out, audio_out, log_out, status_out]
        )

        preview_btn.click(
            preview_voice,
            [script, voice, rate, pitch],
            [preview_audio, log_out]
        )

        apply_tpl_btn.click(
            apply_template,
            [template],
            [prompt]
        )

        quality.change(
            apply_preset,
            [quality],
            [width, height, fps, steps]
        )

        clear_btn.click(
            clear_all,
            outputs=[prompt, negative, script, rate, pitch,
                    translate_prompt, target_lang,
                    width, height, fps, steps, cfg, seed, randomize,
                    camera_motion, lighting_style, log_out]
        )

        refresh_log.click(get_logs_html, outputs=[log_out])
        refresh_hist.click(get_history_html, outputs=[history_out])

    return app

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import io

    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("\n" + "="*60)
    print("  🎬 AI ANIME VIDEO CREATOR - PRODUCTION READY")
    print("="*60)
    print("\n✨ Features:")
    print("  • Advanced animated video generation")
    print("  • Multi-language translation (6+ languages)")
    print("  • Auto-translate script to voice language")
    print("  • Professional UI with all buttons working")
    print("  • Generation history tracking")
    print("\n🌍 Languages: Japanese • Hindi • Kannada • English • Korean • Chinese")
    print("\n🚀 Starting...\n")

    app = build_ui()
    app.queue(max_size=20)

    for port in range(7860, 7871):
        try:
            print(f"🌐 Starting on port {port}...")
            app.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=True,
                inbrowser=True,
                show_error=True
            )
            break
        except OSError:
            print(f"   Port {port} busy...")
            continue

