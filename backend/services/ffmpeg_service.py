"""
FFmpeg Service - Media Processing & Muxing
==========================================
This module handles all video/audio processing using FFmpeg.
Provides video looping, audio muxing, format conversion, and more.
"""

import subprocess
import ffmpeg
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from loguru import logger
import tempfile
import json


@dataclass
class MediaInfo:
    """Information about a media file."""
    duration: float
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    codec: Optional[str]
    format: str
    size_bytes: int


@dataclass
class ProcessResult:
    """Result from a media processing operation."""
    success: bool
    output_path: Optional[Path]
    duration: float
    error_message: Optional[str] = None


class FFmpegService:
    """
    FFmpeg-based media processing service.

    Handles all video/audio manipulation including:
    - Video/audio muxing with loop support
    - Format conversion
    - Video trimming and concatenation
    - Thumbnail generation
    - Media info extraction
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        output_dir: Path = Path("./outputs")
    ):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_ffmpeg(self) -> bool:
        """Verify FFmpeg is installed and accessible."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_media_info(self, file_path: Path) -> Optional[MediaInfo]:
        """Extract detailed information about a media file."""
        try:
            probe = ffmpeg.probe(str(file_path))

            format_info = probe.get('format', {})
            duration = float(format_info.get('duration', 0))
            size_bytes = int(format_info.get('size', 0))
            format_name = format_info.get('format_name', 'unknown')

            # Find video stream
            video_stream = None
            for stream in probe.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            width = None
            height = None
            fps = None
            codec = None

            if video_stream:
                width = video_stream.get('width')
                height = video_stream.get('height')
                codec = video_stream.get('codec_name')

                # Parse frame rate
                fps_str = video_stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den > 0 else 0

            return MediaInfo(
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                codec=codec,
                format=format_name,
                size_bytes=size_bytes
            )

        except Exception as e:
            logger.error(f"Failed to get media info: {e}")
            return None

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of an audio file in seconds."""
        info = self.get_media_info(audio_path)
        return info.duration if info else 0.0

    def mux_audio_video(
        self,
        video_path: Path,
        audio_path: Path,
        output_filename: str,
        loop_video: bool = True,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23
    ) -> ProcessResult:
        """
        Combine video and audio into a single file.

        If the video is shorter than the audio, it will be looped
        (when loop_video=True). The output will match the audio duration.

        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_filename: Name for output file
            loop_video: Whether to loop video if shorter than audio
            video_codec: Video codec to use
            audio_codec: Audio codec to use
            crf: Constant Rate Factor (quality, lower = better)

        Returns:
            ProcessResult with output path and status
        """
        try:
            output_path = self.output_dir / output_filename

            # Get durations
            audio_info = self.get_media_info(audio_path)
            video_info = self.get_media_info(video_path)

            if not audio_info or not video_info:
                return ProcessResult(
                    success=False,
                    output_path=None,
                    duration=0,
                    error_message="Failed to read media info"
                )

            audio_duration = audio_info.duration
            video_duration = video_info.duration

            logger.info(f"Audio duration: {audio_duration:.2f}s, Video duration: {video_duration:.2f}s")

            if loop_video and video_duration < audio_duration:
                # Video needs looping
                input_video = ffmpeg.input(str(video_path), stream_loop=-1)
            else:
                input_video = ffmpeg.input(str(video_path))

            input_audio = ffmpeg.input(str(audio_path))

            # Build output
            output = ffmpeg.output(
                input_video,
                input_audio,
                str(output_path),
                vcodec=video_codec,
                acodec=audio_codec,
                crf=crf,
                shortest=None,  # Stop when shortest stream ends
                pix_fmt='yuv420p'  # Ensure compatibility
            )

            # Run with overwrite
            output = output.overwrite_output()

            logger.info(f"Running FFmpeg mux command...")
            output.run(capture_stdout=True, capture_stderr=True)

            # Verify output
            output_info = self.get_media_info(output_path)

            return ProcessResult(
                success=True,
                output_path=output_path,
                duration=output_info.duration if output_info else audio_duration
            )

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg mux failed: {error_msg}")
            return ProcessResult(
                success=False,
                output_path=None,
                duration=0,
                error_message=error_msg
            )
        except Exception as e:
            logger.error(f"Mux failed: {e}")
            return ProcessResult(
                success=False,
                output_path=None,
                duration=0,
                error_message=str(e)
            )

    def loop_video(
        self,
        video_path: Path,
        target_duration: float,
        output_filename: str
    ) -> ProcessResult:
        """
        Loop a video to reach a target duration.

        Args:
            video_path: Path to source video
            target_duration: Desired duration in seconds
            output_filename: Name for output file
        """
        try:
            output_path = self.output_dir / output_filename

            input_video = ffmpeg.input(str(video_path), stream_loop=-1)

            output = ffmpeg.output(
                input_video,
                str(output_path),
                t=target_duration,
                vcodec='libx264',
                crf=23,
                pix_fmt='yuv420p'
            ).overwrite_output()

            output.run(capture_stdout=True, capture_stderr=True)

            output_info = self.get_media_info(output_path)

            return ProcessResult(
                success=True,
                output_path=output_path,
                duration=output_info.duration if output_info else target_duration
            )

        except Exception as e:
            logger.error(f"Video loop failed: {e}")
            return ProcessResult(
                success=False,
                output_path=None,
                duration=0,
                error_message=str(e)
            )

    def convert_video(
        self,
        input_path: Path,
        output_filename: str,
        format: str = "mp4",
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23,
        fps: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> ProcessResult:
        """
        Convert video to different format/codec with optional resize.

        Args:
            input_path: Source video path
            output_filename: Output filename
            format: Target format (mp4, webm, etc.)
            video_codec: Video codec
            audio_codec: Audio codec
            crf: Quality factor
            fps: Target frame rate (None to keep original)
            width: Target width (None to keep original)
            height: Target height (None to keep original)
        """
        try:
            output_path = self.output_dir / output_filename

            input_video = ffmpeg.input(str(input_path))

            # Build filter chain
            video = input_video.video

            if fps:
                video = video.filter('fps', fps=fps)

            if width and height:
                video = video.filter('scale', width, height)
            elif width:
                video = video.filter('scale', width, -2)
            elif height:
                video = video.filter('scale', -2, height)

            # Output
            output = ffmpeg.output(
                video,
                input_video.audio,
                str(output_path),
                vcodec=video_codec,
                acodec=audio_codec,
                crf=crf,
                pix_fmt='yuv420p'
            ).overwrite_output()

            output.run(capture_stdout=True, capture_stderr=True)

            output_info = self.get_media_info(output_path)

            return ProcessResult(
                success=True,
                output_path=output_path,
                duration=output_info.duration if output_info else 0
            )

        except Exception as e:
            logger.error(f"Video conversion failed: {e}")
            return ProcessResult(
                success=False,
                output_path=None,
                duration=0,
                error_message=str(e)
            )

    def extract_thumbnail(
        self,
        video_path: Path,
        output_filename: str,
        timestamp: float = 0.5,
        width: int = 320
    ) -> Optional[Path]:
        """
        Extract a thumbnail frame from a video.

        Args:
            video_path: Source video path
            output_filename: Output image filename
            timestamp: Time position (0-1 as ratio, or seconds if > 1)
            width: Thumbnail width (height auto-calculated)
        """
        try:
            output_path = self.output_dir / output_filename

            video_info = self.get_media_info(video_path)
            if not video_info:
                return None

            # Calculate timestamp
            if 0 <= timestamp <= 1:
                seek_time = video_info.duration * timestamp
            else:
                seek_time = min(timestamp, video_info.duration)

            input_video = ffmpeg.input(str(video_path), ss=seek_time)

            output = (
                input_video
                .filter('scale', width, -2)
                .output(str(output_path), vframes=1)
                .overwrite_output()
            )

            output.run(capture_stdout=True, capture_stderr=True)

            return output_path if output_path.exists() else None

        except Exception as e:
            logger.error(f"Thumbnail extraction failed: {e}")
            return None

    def concatenate_videos(
        self,
        video_paths: List[Path],
        output_filename: str
    ) -> ProcessResult:
        """
        Concatenate multiple videos into one.

        All videos should have the same codec/resolution for best results.
        """
        try:
            output_path = self.output_dir / output_filename

            # Create concat file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                delete=False
            ) as f:
                for vpath in video_paths:
                    f.write(f"file '{vpath.absolute()}'\n")
                concat_file = f.name

            # Run concat
            input_stream = ffmpeg.input(concat_file, format='concat', safe=0)
            output = ffmpeg.output(
                input_stream,
                str(output_path),
                c='copy'
            ).overwrite_output()

            output.run(capture_stdout=True, capture_stderr=True)

            # Cleanup
            Path(concat_file).unlink()

            output_info = self.get_media_info(output_path)

            return ProcessResult(
                success=True,
                output_path=output_path,
                duration=output_info.duration if output_info else 0
            )

        except Exception as e:
            logger.error(f"Video concatenation failed: {e}")
            return ProcessResult(
                success=False,
                output_path=None,
                duration=0,
                error_message=str(e)
            )

    def add_subtitles(
        self,
        video_path: Path,
        subtitle_path: Path,
        output_filename: str,
        font_size: int = 24,
        font_color: str = "white",
        outline_color: str = "black"
    ) -> ProcessResult:
        """
        Burn subtitles into a video.

        Args:
            video_path: Source video
            subtitle_path: Path to SRT or ASS subtitle file
            output_filename: Output filename
            font_size: Subtitle font size
            font_color: Subtitle text color
            outline_color: Subtitle outline color
        """
        try:
            output_path = self.output_dir / output_filename

            input_video = ffmpeg.input(str(video_path))

            # Build subtitle filter
            sub_filter = (
                f"subtitles={subtitle_path}:"
                f"force_style='FontSize={font_size},"
                f"PrimaryColour=&H{self._color_to_ass(font_color)},"
                f"OutlineColour=&H{self._color_to_ass(outline_color)},"
                f"BorderStyle=3,Outline=2'"
            )

            output = (
                input_video
                .output(
                    str(output_path),
                    vf=sub_filter,
                    vcodec='libx264',
                    acodec='copy',
                    crf=23
                )
                .overwrite_output()
            )

            output.run(capture_stdout=True, capture_stderr=True)

            output_info = self.get_media_info(output_path)

            return ProcessResult(
                success=True,
                output_path=output_path,
                duration=output_info.duration if output_info else 0
            )

        except Exception as e:
            logger.error(f"Subtitle burn failed: {e}")
            return ProcessResult(
                success=False,
                output_path=None,
                duration=0,
                error_message=str(e)
            )

    def _color_to_ass(self, color: str) -> str:
        """Convert color name to ASS format (BGR hex)."""
        colors = {
            "white": "FFFFFF",
            "black": "000000",
            "red": "0000FF",
            "green": "00FF00",
            "blue": "FF0000",
            "yellow": "00FFFF"
        }
        return colors.get(color.lower(), "FFFFFF")

    def create_gif(
        self,
        video_path: Path,
        output_filename: str,
        fps: int = 10,
        width: int = 480,
        duration: Optional[float] = None
    ) -> ProcessResult:
        """
        Convert video to GIF.

        Args:
            video_path: Source video
            output_filename: Output GIF filename
            fps: GIF frame rate
            width: GIF width
            duration: Max duration (None for full video)
        """
        try:
            output_path = self.output_dir / output_filename

            input_kwargs = {}
            if duration:
                input_kwargs['t'] = duration

            input_video = ffmpeg.input(str(video_path), **input_kwargs)

            # Generate palette for better quality
            palette_path = self.output_dir / "palette.png"

            # Generate palette
            palette_gen = (
                input_video
                .filter('fps', fps=fps)
                .filter('scale', width, -1, flags='lanczos')
                .filter('palettegen')
                .output(str(palette_path))
                .overwrite_output()
            )
            palette_gen.run(capture_stdout=True, capture_stderr=True)

            # Use palette to create GIF
            input_video = ffmpeg.input(str(video_path), **input_kwargs)
            palette = ffmpeg.input(str(palette_path))

            output = (
                ffmpeg.filter(
                    [
                        input_video.filter('fps', fps=fps).filter('scale', width, -1, flags='lanczos'),
                        palette
                    ],
                    'paletteuse'
                )
                .output(str(output_path))
                .overwrite_output()
            )

            output.run(capture_stdout=True, capture_stderr=True)

            # Cleanup palette
            palette_path.unlink(missing_ok=True)

            output_info = self.get_media_info(output_path)

            return ProcessResult(
                success=True,
                output_path=output_path,
                duration=output_info.duration if output_info else 0
            )

        except Exception as e:
            logger.error(f"GIF creation failed: {e}")
            return ProcessResult(
                success=False,
                output_path=None,
                duration=0,
                error_message=str(e)
            )

