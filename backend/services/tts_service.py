"""
Text-to-Speech Service - Edge-TTS Integration
==============================================
This module handles audio synthesis using Microsoft Edge TTS.
Provides high-quality anime-style voices with zero GPU overhead.
"""

import asyncio
import edge_tts
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from loguru import logger
import aiofiles
from pydub import AudioSegment
import tempfile


@dataclass
class VoiceInfo:
    """Information about an available TTS voice."""
    id: str
    name: str
    gender: str
    locale: str
    style: Optional[str] = None


@dataclass
class AudioResult:
    """Result from audio generation."""
    success: bool
    file_path: Optional[Path]
    duration_seconds: float
    sample_rate: int
    error_message: Optional[str] = None


class TTSService:
    """
    Text-to-Speech service using Microsoft Edge Neural TTS.

    This service provides high-quality voice synthesis optimized for
    anime-style content, with support for Japanese and English voices.
    """

    # Curated list of anime-suitable voices
    ANIME_VOICES = {
        # Japanese Female (Energetic)
        "ja-JP-NanamiNeural": VoiceInfo(
            id="ja-JP-NanamiNeural",
            name="Nanami",
            gender="Female",
            locale="ja-JP",
            style="Energetic Anime Girl"
        ),
        # Japanese Female (Gentle)
        "ja-JP-AoiNeural": VoiceInfo(
            id="ja-JP-AoiNeural",
            name="Aoi",
            gender="Female",
            locale="ja-JP",
            style="Gentle & Soft"
        ),
        # Japanese Male (Young)
        "ja-JP-KeitaNeural": VoiceInfo(
            id="ja-JP-KeitaNeural",
            name="Keita",
            gender="Male",
            locale="ja-JP",
            style="Young & Dynamic"
        ),
        # Japanese Male (Mature)
        "ja-JP-DaichiNeural": VoiceInfo(
            id="ja-JP-DaichiNeural",
            name="Daichi",
            gender="Male",
            locale="ja-JP",
            style="Deep & Confident"
        ),
        # English Female (Narrator)
        "en-US-AriaNeural": VoiceInfo(
            id="en-US-AriaNeural",
            name="Aria",
            gender="Female",
            locale="en-US",
            style="Narrator Style"
        ),
        # English Female (Friendly)
        "en-US-JennyNeural": VoiceInfo(
            id="en-US-JennyNeural",
            name="Jenny",
            gender="Female",
            locale="en-US",
            style="Friendly & Warm"
        ),
        # English Male
        "en-US-GuyNeural": VoiceInfo(
            id="en-US-GuyNeural",
            name="Guy",
            gender="Male",
            locale="en-US",
            style="Professional"
        ),
        # Korean Female
        "ko-KR-SunHiNeural": VoiceInfo(
            id="ko-KR-SunHiNeural",
            name="SunHi",
            gender="Female",
            locale="ko-KR",
            style="K-Drama Style"
        ),
        # Chinese Female
        "zh-CN-XiaoxiaoNeural": VoiceInfo(
            id="zh-CN-XiaoxiaoNeural",
            name="Xiaoxiao",
            gender="Female",
            locale="zh-CN",
            style="Warm & Expressive"
        ),
        # Hindi Female (Expressive)
        "hi-IN-SwaraNeural": VoiceInfo(
            id="hi-IN-SwaraNeural",
            name="Swara",
            gender="Female",
            locale="hi-IN",
            style="Expressive & Warm"
        ),
        # Hindi Male (Professional)
        "hi-IN-MadhurNeural": VoiceInfo(
            id="hi-IN-MadhurNeural",
            name="Madhur",
            gender="Male",
            locale="hi-IN",
            style="Professional & Clear"
        ),
        # Kannada Female (Melodic)
        "kn-IN-SapnaNeural": VoiceInfo(
            id="kn-IN-SapnaNeural",
            name="Sapna",
            gender="Female",
            locale="kn-IN",
            style="Melodic & Pleasant"
        ),
        # Kannada Male (Deep)
        "kn-IN-GaganNeural": VoiceInfo(
            id="kn-IN-GaganNeural",
            name="Gagan",
            gender="Male",
            locale="kn-IN",
            style="Deep & Authoritative"
        ),
    }

    def __init__(self, output_dir: Path = Path("./temp")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_available_voices(cls) -> List[VoiceInfo]:
        """Get list of curated anime-suitable voices."""
        return list(cls.ANIME_VOICES.values())

    @classmethod
    def get_voice_choices(cls) -> List[tuple[str, str]]:
        """Get voice choices for Gradio dropdown."""
        choices = []
        for voice in cls.ANIME_VOICES.values():
            label = f"{voice.name} ({voice.locale}) - {voice.style}"
            choices.append((label, voice.id))
        return choices

    @classmethod
    async def fetch_all_voices(cls) -> List[Dict[str, Any]]:
        """Fetch all available voices from Edge TTS."""
        voices = await edge_tts.VoicesManager.create()
        return voices.voices

    async def generate_audio(
        self,
        text: str,
        voice_id: str = "ja-JP-NanamiNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        output_filename: Optional[str] = None
    ) -> AudioResult:
        """
        Generate audio from text using Edge TTS.

        Args:
            text: The text to convert to speech
            voice_id: The voice identifier to use
            rate: Speech rate adjustment (e.g., "+10%", "-5%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz")
            volume: Volume adjustment (e.g., "+10%")
            output_filename: Optional custom filename

        Returns:
            AudioResult with file path and duration
        """
        try:
            # Generate unique filename if not provided
            if output_filename is None:
                import uuid
                output_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"

            output_path = self.output_dir / output_filename

            # Create TTS communicate instance
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate=rate,
                pitch=pitch,
                volume=volume
            )

            # Generate and save audio
            await communicate.save(str(output_path))

            # Get audio duration using pydub
            audio = AudioSegment.from_mp3(str(output_path))
            duration_seconds = len(audio) / 1000.0

            logger.info(f"Generated audio: {output_path}, duration: {duration_seconds:.2f}s")

            return AudioResult(
                success=True,
                file_path=output_path,
                duration_seconds=duration_seconds,
                sample_rate=audio.frame_rate
            )

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return AudioResult(
                success=False,
                file_path=None,
                duration_seconds=0,
                sample_rate=0,
                error_message=str(e)
            )

    def generate_audio_sync(
        self,
        text: str,
        voice_id: str = "ja-JP-NanamiNeural",
        **kwargs
    ) -> AudioResult:
        """
        Synchronous wrapper for audio generation.

        Use this when calling from non-async context (e.g., Gradio callbacks).
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_audio(text, voice_id, **kwargs)
            )
        finally:
            loop.close()

    async def generate_with_subtitles(
        self,
        text: str,
        voice_id: str = "ja-JP-NanamiNeural",
        output_filename: Optional[str] = None
    ) -> tuple[AudioResult, List[Dict[str, Any]]]:
        """
        Generate audio with word-level timing for subtitles.

        Returns audio result and list of subtitle entries with timing.
        """
        try:
            if output_filename is None:
                import uuid
                output_filename = f"tts_{uuid.uuid4().hex[:8]}.mp3"

            output_path = self.output_dir / output_filename
            subtitles = []

            communicate = edge_tts.Communicate(text=text, voice=voice_id)

            # Collect audio and subtitle data
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
                elif chunk["type"] == "WordBoundary":
                    subtitles.append({
                        "text": chunk["text"],
                        "start": chunk["offset"] / 10000000,  # Convert to seconds
                        "duration": chunk["duration"] / 10000000
                    })

            # Save audio
            async with aiofiles.open(output_path, 'wb') as f:
                await f.write(audio_data)

            # Get duration
            audio = AudioSegment.from_mp3(str(output_path))
            duration_seconds = len(audio) / 1000.0

            result = AudioResult(
                success=True,
                file_path=output_path,
                duration_seconds=duration_seconds,
                sample_rate=audio.frame_rate
            )

            return result, subtitles

        except Exception as e:
            logger.error(f"TTS with subtitles failed: {e}")
            return AudioResult(
                success=False,
                file_path=None,
                duration_seconds=0,
                sample_rate=0,
                error_message=str(e)
            ), []

    async def concatenate_audio(
        self,
        audio_files: List[Path],
        output_filename: str,
        gap_ms: int = 500
    ) -> AudioResult:
        """
        Concatenate multiple audio files with optional gaps.

        Args:
            audio_files: List of audio file paths to concatenate
            output_filename: Name for the output file
            gap_ms: Milliseconds of silence between clips
        """
        try:
            combined = AudioSegment.empty()
            silence = AudioSegment.silent(duration=gap_ms)

            for i, audio_file in enumerate(audio_files):
                segment = AudioSegment.from_file(str(audio_file))
                combined += segment
                if i < len(audio_files) - 1:
                    combined += silence

            output_path = self.output_dir / output_filename
            combined.export(str(output_path), format="mp3")

            return AudioResult(
                success=True,
                file_path=output_path,
                duration_seconds=len(combined) / 1000.0,
                sample_rate=combined.frame_rate
            )

        except Exception as e:
            logger.error(f"Audio concatenation failed: {e}")
            return AudioResult(
                success=False,
                file_path=None,
                duration_seconds=0,
                sample_rate=0,
                error_message=str(e)
            )

    @staticmethod
    def calculate_frame_count(
        audio_duration: float,
        fps: int = 15
    ) -> int:
        """
        Calculate required video frames based on audio duration.

        This is the core of the "Audio-First" pipeline design.

        Args:
            audio_duration: Duration of audio in seconds
            fps: Target frames per second

        Returns:
            Number of frames needed to match audio duration
        """
        import math
        return math.ceil(audio_duration * fps)


# Convenience function for quick audio generation
async def quick_generate(
    text: str,
    voice: str = "ja-JP-NanamiNeural",
    output_dir: str = "./temp"
) -> Path:
    """Quick one-liner for generating audio."""
    service = TTSService(output_dir=Path(output_dir))
    result = await service.generate_audio(text, voice)
    if result.success:
        return result.file_path
    raise RuntimeError(result.error_message)

