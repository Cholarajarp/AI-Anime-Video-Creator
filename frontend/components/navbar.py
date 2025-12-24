"""
Navigation Bar Component
========================
Top navigation bar with app title, status indicators, and theme toggle.
"""

import gradio as gr
from typing import Callable, Optional


def create_navbar(
    app_name: str = "Anime Video Creator",
    version: str = "1.0.0",
    on_theme_toggle: Optional[Callable] = None
) -> tuple:
    """
    Create the navigation bar component.

    Returns:
        Tuple of (navbar container, status indicator, connection status)
    """
    with gr.Row(elem_classes=["navbar"]) as navbar:
        # Logo and Title
        with gr.Column(scale=1, min_width=200):
            gr.Markdown(
                f"""
                <div class="navbar-brand">
                    <span class="navbar-logo">🎬</span>
                    <span class="navbar-title">{app_name}</span>
                    <span class="navbar-version">v{version}</span>
                </div>
                """,
                elem_classes=["navbar-brand-container"]
            )

        # Center - Navigation Links
        with gr.Column(scale=2):
            with gr.Row():
                workspace_btn = gr.Button(
                    "🎨 Workspace",
                    variant="secondary",
                    size="sm",
                    elem_classes=["nav-link", "active"]
                )
                queue_btn = gr.Button(
                    "📋 Queue",
                    variant="secondary",
                    size="sm",
                    elem_classes=["nav-link"]
                )
                gallery_btn = gr.Button(
                    "🖼️ Gallery",
                    variant="secondary",
                    size="sm",
                    elem_classes=["nav-link"]
                )
                settings_btn = gr.Button(
                    "⚙️ Settings",
                    variant="secondary",
                    size="sm",
                    elem_classes=["nav-link"]
                )

        # Right - Status and Controls
        with gr.Column(scale=1, min_width=300):
            with gr.Row():
                # Connection Status
                connection_status = gr.HTML(
                    """
                    <div class="connection-status">
                        <span class="status-dot connected"></span>
                        <span class="status-text">ComfyUI Connected</span>
                    </div>
                    """,
                    elem_classes=["connection-indicator"]
                )

                # GPU Status
                gpu_status = gr.HTML(
                    """
                    <div class="gpu-status">
                        <span class="gpu-icon">🎮</span>
                        <span class="gpu-text">GPU: NVIDIA RTX</span>
                        <span class="vram-bar">
                            <span class="vram-used" style="width: 45%;"></span>
                        </span>
                    </div>
                    """,
                    elem_classes=["gpu-indicator"]
                )

                # Theme Toggle
                theme_toggle = gr.Button(
                    "🌙",
                    variant="secondary",
                    size="sm",
                    elem_classes=["theme-toggle"]
                )

    return navbar, connection_status, gpu_status, theme_toggle, {
        "workspace": workspace_btn,
        "queue": queue_btn,
        "gallery": gallery_btn,
        "settings": settings_btn
    }


def create_status_bar() -> tuple:
    """
    Create the bottom status bar.

    Returns:
        Tuple of (status bar, queue count, processing status)
    """
    with gr.Row(elem_classes=["status-bar"]) as status_bar:
        # Left - Current Operation
        with gr.Column(scale=2):
            current_operation = gr.HTML(
                """
                <div class="current-operation">
                    <span class="operation-icon">⏳</span>
                    <span class="operation-text">Ready</span>
                </div>
                """,
                elem_classes=["operation-status"]
            )

        # Center - Progress
        with gr.Column(scale=3):
            with gr.Row():
                progress_bar = gr.HTML(
                    """
                    <div class="global-progress">
                        <div class="progress-track">
                            <div class="progress-fill" style="width: 0%;"></div>
                        </div>
                        <span class="progress-text">0%</span>
                    </div>
                    """,
                    elem_classes=["progress-container"]
                )

        # Right - Queue Info
        with gr.Column(scale=1):
            queue_info = gr.HTML(
                """
                <div class="queue-info">
                    <span class="queue-icon">📋</span>
                    <span class="queue-text">Queue: 0</span>
                    <span class="separator">|</span>
                    <span class="running-text">Running: 0</span>
                </div>
                """,
                elem_classes=["queue-status"]
            )

    return status_bar, current_operation, progress_bar, queue_info


def update_connection_status(connected: bool, gpu_info: Optional[dict] = None) -> tuple:
    """
    Update connection and GPU status HTML.

    Args:
        connected: Whether ComfyUI is connected
        gpu_info: Optional GPU information dict

    Returns:
        Tuple of (connection HTML, GPU HTML)
    """
    connection_class = "connected" if connected else "disconnected"
    connection_text = "ComfyUI Connected" if connected else "ComfyUI Disconnected"

    connection_html = f"""
    <div class="connection-status">
        <span class="status-dot {connection_class}"></span>
        <span class="status-text">{connection_text}</span>
    </div>
    """

    if gpu_info:
        gpu_name = gpu_info.get("name", "Unknown GPU")
        vram_used = gpu_info.get("vram_used", 0)
        vram_total = gpu_info.get("vram_total", 1)
        vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0

        gpu_html = f"""
        <div class="gpu-status">
            <span class="gpu-icon">🎮</span>
            <span class="gpu-text">{gpu_name}</span>
            <span class="vram-bar">
                <span class="vram-used" style="width: {vram_percent:.1f}%;"></span>
            </span>
            <span class="vram-text">{vram_used:.1f}/{vram_total:.1f} GB</span>
        </div>
        """
    else:
        gpu_html = """
        <div class="gpu-status">
            <span class="gpu-icon">❌</span>
            <span class="gpu-text">No GPU</span>
        </div>
        """

    return connection_html, gpu_html


def update_progress_bar(progress: float, message: str = "") -> str:
    """
    Update the global progress bar HTML.

    Args:
        progress: Progress value 0-1
        message: Optional progress message

    Returns:
        Progress bar HTML
    """
    percent = int(progress * 100)
    display_text = message if message else f"{percent}%"

    return f"""
    <div class="global-progress">
        <div class="progress-track">
            <div class="progress-fill" style="width: {percent}%;"></div>
        </div>
        <span class="progress-text">{display_text}</span>
    </div>
    """


def update_queue_info(queue_length: int, running: int) -> str:
    """
    Update queue info HTML.

    Args:
        queue_length: Number of jobs in queue
        running: Number of currently running jobs

    Returns:
        Queue info HTML
    """
    return f"""
    <div class="queue-info">
        <span class="queue-icon">📋</span>
        <span class="queue-text">Queue: {queue_length}</span>
        <span class="separator">|</span>
        <span class="running-text">Running: {running}</span>
    </div>
    """

