"""
Video Workspace Component
=========================
Main workspace with video preview, progress, and output management.
"""

import gradio as gr
from typing import Optional, Dict, Any, List
from pathlib import Path


def create_video_workspace() -> dict:
    """
    Create the video workspace component.

    Returns:
        Dictionary of workspace components
    """
    with gr.Column(elem_classes=["video-workspace"]) as workspace:
        # Header with current job info
        with gr.Row(elem_classes=["workspace-header"]):
            current_job_info = gr.HTML(
                render_no_job_state(),
                elem_classes=["current-job-info"]
            )

        # Main Preview Area
        with gr.Row(elem_classes=["preview-area"]):
            with gr.Column(scale=3):
                # Video Player
                video_preview = gr.Video(
                    label="Generated Video",
                    elem_classes=["main-video-player"],
                    interactive=False,
                    show_download_button=True,
                    height=400
                )

                # Preview Controls
                with gr.Row(elem_classes=["preview-controls"]):
                    play_btn = gr.Button("▶️ Play", size="sm")
                    loop_btn = gr.Button("🔁 Loop", size="sm")
                    fullscreen_btn = gr.Button("⛶ Fullscreen", size="sm")

            with gr.Column(scale=1):
                # Thumbnail / Frame Preview
                frame_preview = gr.Image(
                    label="Thumbnail",
                    elem_classes=["frame-preview"],
                    height=200,
                    interactive=False
                )

                # Audio Player
                audio_preview = gr.Audio(
                    label="Audio Track",
                    elem_classes=["audio-player"],
                    interactive=False
                )

        # Progress Section
        with gr.Row(elem_classes=["progress-section"]):
            with gr.Column():
                progress_bar = gr.HTML(
                    render_progress_bar(0, "Ready to generate"),
                    elem_classes=["generation-progress"]
                )

                progress_details = gr.HTML(
                    "",
                    elem_classes=["progress-details"]
                )

        # Generation Info
        with gr.Accordion("📊 Generation Details", open=False):
            generation_info = gr.HTML(
                render_empty_info(),
                elem_classes=["generation-info"]
            )

        # Action Buttons
        with gr.Row(elem_classes=["output-actions"]):
            download_video_btn = gr.Button(
                "⬇️ Download Video",
                variant="primary",
                interactive=False
            )
            download_audio_btn = gr.Button(
                "⬇️ Download Audio",
                variant="secondary",
                interactive=False
            )
            regenerate_btn = gr.Button(
                "🔄 Regenerate",
                variant="secondary",
                interactive=False
            )
            share_btn = gr.Button(
                "📤 Share",
                variant="secondary",
                interactive=False
            )

    return {
        "workspace": workspace,
        "current_job_info": current_job_info,
        "video_preview": video_preview,
        "frame_preview": frame_preview,
        "audio_preview": audio_preview,
        "progress_bar": progress_bar,
        "progress_details": progress_details,
        "generation_info": generation_info,
        "play_btn": play_btn,
        "loop_btn": loop_btn,
        "fullscreen_btn": fullscreen_btn,
        "download_video_btn": download_video_btn,
        "download_audio_btn": download_audio_btn,
        "regenerate_btn": regenerate_btn,
        "share_btn": share_btn
    }


def render_no_job_state() -> str:
    """Render the state when no job is active."""
    return """
    <div class="no-job-state">
        <div class="no-job-icon">🎬</div>
        <div class="no-job-text">
            <h3>Ready to Create</h3>
            <p>Configure your settings and click "Generate Video" to start</p>
        </div>
    </div>
    """


def render_current_job(job: Dict[str, Any]) -> str:
    """Render current job information."""
    job_id = job.get("id", "unknown")[:8]
    status = job.get("status", "unknown")
    prompt = job.get("prompt", "")[:80] + "..." if len(job.get("prompt", "")) > 80 else job.get("prompt", "")

    status_badges = {
        "pending": "⏳ Pending",
        "queued": "📋 Queued",
        "running": "🔄 Generating",
        "completed": "✅ Completed",
        "failed": "❌ Failed",
        "cancelled": "⛔ Cancelled"
    }

    status_text = status_badges.get(status, status)

    return f"""
    <div class="current-job-header">
        <div class="job-id-badge">Job #{job_id}</div>
        <div class="job-status-badge status-{status}">{status_text}</div>
    </div>
    <div class="job-prompt-preview">{prompt}</div>
    """


def render_progress_bar(progress: float, message: str = "") -> str:
    """
    Render the generation progress bar.

    Args:
        progress: Progress value 0-1
        message: Progress message
    """
    percent = int(progress * 100)

    # Determine color based on progress
    if percent < 25:
        color = "#3b82f6"  # Blue
    elif percent < 50:
        color = "#8b5cf6"  # Purple
    elif percent < 75:
        color = "#f59e0b"  # Orange
    else:
        color = "#10b981"  # Green

    return f"""
    <div class="progress-container">
        <div class="progress-header">
            <span class="progress-label">{message or "Generating..."}</span>
            <span class="progress-percent">{percent}%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width: {percent}%; background: {color};"></div>
        </div>
    </div>
    """


def render_progress_steps(current_step: int, total_steps: int = 5) -> str:
    """
    Render step-by-step progress indicators.

    Args:
        current_step: Current step (1-indexed)
        total_steps: Total number of steps
    """
    steps = [
        ("🎤", "Generate Audio"),
        ("📐", "Calculate Frames"),
        ("🎨", "Generate Video"),
        ("🎬", "Process Video"),
        ("✨", "Finalize")
    ]

    html_parts = []
    for i, (icon, label) in enumerate(steps[:total_steps], 1):
        if i < current_step:
            status_class = "completed"
        elif i == current_step:
            status_class = "active"
        else:
            status_class = "pending"

        html_parts.append(f"""
        <div class="step-item {status_class}">
            <div class="step-icon">{icon}</div>
            <div class="step-label">{label}</div>
        </div>
        """)

    return f"""
    <div class="progress-steps">
        {"".join(html_parts)}
    </div>
    """


def render_empty_info() -> str:
    """Render empty generation info state."""
    return """
    <div class="info-empty">
        Generation details will appear here after completion
    </div>
    """


def render_generation_info(job: Dict[str, Any]) -> str:
    """Render detailed generation information."""
    info_items = [
        ("🎬", "Duration", f"{job.get('audio_duration', 0):.2f}s"),
        ("🖼️", "Frames", str(job.get('frame_count', 0))),
        ("📐", "Resolution", f"{job.get('width', 512)}x{job.get('height', 512)}"),
        ("⚡", "FPS", str(job.get('fps', 15))),
        ("🔢", "Steps", str(job.get('steps', 20))),
        ("🎯", "CFG Scale", str(job.get('cfg_scale', 7.0))),
        ("🎲", "Seed", str(job.get('seed', -1))),
        ("🤖", "Model", job.get('checkpoint', 'Unknown')[:30]),
        ("⏱️", "Gen Time", f"{job.get('generation_time', 0):.2f}s"),
    ]

    html_items = []
    for icon, label, value in info_items:
        html_items.append(f"""
        <div class="info-item">
            <span class="info-icon">{icon}</span>
            <span class="info-label">{label}:</span>
            <span class="info-value">{value}</span>
        </div>
        """)

    return f"""
    <div class="generation-info-grid">
        {"".join(html_items)}
    </div>
    """


def create_gallery() -> dict:
    """
    Create the video gallery component.

    Returns:
        Dictionary of gallery components
    """
    with gr.Column(elem_classes=["gallery-panel"]) as panel:
        gr.Markdown(
            """
            ## 🖼️ Gallery
            Your generated videos
            """,
            elem_classes=["panel-header"]
        )

        # Filter and Sort
        with gr.Row():
            sort_by = gr.Dropdown(
                choices=[
                    ("Newest First", "newest"),
                    ("Oldest First", "oldest"),
                    ("Longest First", "longest"),
                    ("Shortest First", "shortest")
                ],
                value="newest",
                label="Sort By",
                scale=1
            )

            filter_status = gr.Dropdown(
                choices=[
                    ("All", "all"),
                    ("Completed", "completed"),
                    ("Failed", "failed")
                ],
                value="all",
                label="Status",
                scale=1
            )

        # Gallery Grid
        gallery = gr.Gallery(
            label="Generated Videos",
            show_label=False,
            columns=3,
            rows=2,
            height=400,
            object_fit="cover",
            elem_classes=["video-gallery"]
        )

        # Selected Video Details
        selected_info = gr.HTML(
            "",
            elem_classes=["selected-video-info"]
        )

        # Pagination
        with gr.Row():
            prev_page_btn = gr.Button("← Previous", size="sm")
            page_info = gr.HTML(
                "<span class='page-info'>Page 1 of 1</span>"
            )
            next_page_btn = gr.Button("Next →", size="sm")

    return {
        "panel": panel,
        "gallery": gallery,
        "sort_by": sort_by,
        "filter_status": filter_status,
        "selected_info": selected_info,
        "prev_page_btn": prev_page_btn,
        "page_info": page_info,
        "next_page_btn": next_page_btn
    }

