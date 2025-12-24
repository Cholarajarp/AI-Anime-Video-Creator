"""
Preview Component
=================
Image and video preview with frame-by-frame navigation.
"""

import gradio as gr
from typing import Optional, List
from pathlib import Path


def create_preview_panel() -> dict:
    """
    Create the preview panel component.

    Returns:
        Dictionary of preview components
    """
    with gr.Column(elem_classes=["preview-panel"]) as panel:
        # Preview Header
        with gr.Row():
            gr.Markdown(
                """
                ## 👁️ Preview
                """,
                elem_classes=["panel-header"]
            )

            preview_mode = gr.Radio(
                choices=["Video", "Frames"],
                value="Video",
                label="Mode",
                elem_classes=["preview-mode-toggle"]
            )

        # Video Preview
        video_preview = gr.Video(
            label="Video Preview",
            visible=True,
            elem_classes=["preview-video"],
            height=300,
            interactive=False
        )

        # Frame Gallery (for frame-by-frame viewing)
        frame_gallery = gr.Gallery(
            label="Frames",
            visible=False,
            columns=4,
            rows=2,
            height=300,
            object_fit="contain",
            elem_classes=["frame-gallery"]
        )

        # Selected Frame
        selected_frame = gr.Image(
            label="Selected Frame",
            visible=False,
            height=200,
            elem_classes=["selected-frame"]
        )

        # Frame Navigation
        with gr.Row(visible=False) as frame_nav:
            prev_frame_btn = gr.Button("◀", size="sm")
            frame_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=1,
                label="Frame",
                interactive=True
            )
            next_frame_btn = gr.Button("▶", size="sm")

        # Preview Info
        preview_info = gr.HTML(
            "",
            elem_classes=["preview-info"]
        )

        # Quick Actions
        with gr.Row(elem_classes=["preview-actions"]):
            save_frame_btn = gr.Button(
                "💾 Save Frame",
                size="sm",
                interactive=False
            )
            compare_btn = gr.Button(
                "🔄 Compare",
                size="sm",
                interactive=False
            )

    return {
        "panel": panel,
        "preview_mode": preview_mode,
        "video_preview": video_preview,
        "frame_gallery": frame_gallery,
        "selected_frame": selected_frame,
        "frame_nav": frame_nav,
        "prev_frame_btn": prev_frame_btn,
        "frame_slider": frame_slider,
        "next_frame_btn": next_frame_btn,
        "preview_info": preview_info,
        "save_frame_btn": save_frame_btn,
        "compare_btn": compare_btn
    }


def toggle_preview_mode(mode: str) -> tuple:
    """
    Toggle between video and frame preview modes.

    Args:
        mode: "Video" or "Frames"

    Returns:
        Visibility states for components
    """
    if mode == "Video":
        return (
            gr.update(visible=True),   # video_preview
            gr.update(visible=False),  # frame_gallery
            gr.update(visible=False),  # selected_frame
            gr.update(visible=False)   # frame_nav
        )
    else:
        return (
            gr.update(visible=False),  # video_preview
            gr.update(visible=True),   # frame_gallery
            gr.update(visible=True),   # selected_frame
            gr.update(visible=True)    # frame_nav
        )


def render_preview_info(
    video_path: Optional[str] = None,
    duration: float = 0,
    fps: int = 15,
    resolution: tuple = (512, 512),
    frame_count: int = 0
) -> str:
    """
    Render preview information HTML.

    Args:
        video_path: Path to video file
        duration: Video duration in seconds
        fps: Frames per second
        resolution: (width, height) tuple
        frame_count: Total number of frames
    """
    if not video_path:
        return """
        <div class="preview-info-empty">
            No video loaded
        </div>
        """

    return f"""
    <div class="preview-info-container">
        <div class="info-row">
            <span class="info-label">Duration:</span>
            <span class="info-value">{duration:.2f}s</span>
        </div>
        <div class="info-row">
            <span class="info-label">Resolution:</span>
            <span class="info-value">{resolution[0]}x{resolution[1]}</span>
        </div>
        <div class="info-row">
            <span class="info-label">FPS:</span>
            <span class="info-value">{fps}</span>
        </div>
        <div class="info-row">
            <span class="info-label">Frames:</span>
            <span class="info-value">{frame_count}</span>
        </div>
    </div>
    """


def create_comparison_view() -> dict:
    """
    Create a side-by-side comparison view.

    Returns:
        Dictionary of comparison components
    """
    with gr.Row(elem_classes=["comparison-view"]) as comparison:
        with gr.Column():
            gr.Markdown("### Original")
            original_video = gr.Video(
                label="Original",
                height=250
            )

        with gr.Column():
            gr.Markdown("### New")
            new_video = gr.Video(
                label="New Generation",
                height=250
            )

    return {
        "comparison": comparison,
        "original_video": original_video,
        "new_video": new_video
    }

