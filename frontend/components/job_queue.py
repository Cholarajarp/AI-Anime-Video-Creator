"""
Job Queue Component
===================
Displays and manages the job queue with real-time updates.
"""

import gradio as gr
from typing import List, Dict, Any, Optional
from datetime import datetime


def create_job_queue() -> dict:
    """
    Create the job queue component.

    Returns:
        Dictionary of queue components
    """
    with gr.Column(elem_classes=["job-queue-panel"]) as panel:
        # Header
        with gr.Row():
            gr.Markdown(
                """
                ## 📋 Job Queue
                """,
                elem_classes=["panel-header"]
            )
            refresh_btn = gr.Button(
                "🔄 Refresh",
                size="sm",
                variant="secondary"
            )

        # Queue Stats
        queue_stats = gr.HTML(
            render_queue_stats(0, 0, 2),
            elem_classes=["queue-stats"]
        )

        # Running Jobs Section
        with gr.Accordion("🔄 Running Jobs", open=True):
            running_jobs = gr.HTML(
                render_empty_state("No jobs currently running"),
                elem_classes=["running-jobs-list"]
            )

        # Pending Jobs Section
        with gr.Accordion("⏳ Pending Jobs", open=True):
            pending_jobs = gr.HTML(
                render_empty_state("No jobs in queue"),
                elem_classes=["pending-jobs-list"]
            )

        # Queue Actions
        with gr.Row(elem_classes=["queue-actions"]):
            clear_completed_btn = gr.Button(
                "🗑️ Clear Completed",
                size="sm",
                variant="secondary"
            )
            cancel_all_btn = gr.Button(
                "⛔ Cancel All",
                size="sm",
                variant="stop"
            )

    return {
        "panel": panel,
        "queue_stats": queue_stats,
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "refresh_btn": refresh_btn,
        "clear_completed_btn": clear_completed_btn,
        "cancel_all_btn": cancel_all_btn
    }


def render_queue_stats(
    queue_length: int,
    running: int,
    max_concurrent: int
) -> str:
    """Render queue statistics HTML."""
    return f"""
    <div class="queue-stats-container">
        <div class="stat-item">
            <span class="stat-value">{queue_length}</span>
            <span class="stat-label">In Queue</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">{running}</span>
            <span class="stat-label">Running</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">{max_concurrent}</span>
            <span class="stat-label">Max Parallel</span>
        </div>
    </div>
    """


def render_empty_state(message: str) -> str:
    """Render empty state HTML."""
    return f"""
    <div class="empty-state">
        <span class="empty-icon">📭</span>
        <span class="empty-text">{message}</span>
    </div>
    """


def render_job_card(job: Dict[str, Any], show_actions: bool = True) -> str:
    """
    Render a single job card HTML.

    Args:
        job: Job data dictionary
        show_actions: Whether to show action buttons
    """
    job_id = job.get("id", "unknown")
    prompt = job.get("prompt", "No prompt")[:100] + "..." if len(job.get("prompt", "")) > 100 else job.get("prompt", "No prompt")
    status = job.get("status", "unknown")
    progress = job.get("progress", 0) * 100
    progress_message = job.get("progress_message", "")
    priority = job.get("priority", "normal")
    created_at = job.get("created_at", "")

    # Status styling
    status_colors = {
        "pending": "#6b7280",
        "queued": "#3b82f6",
        "running": "#f59e0b",
        "completed": "#10b981",
        "failed": "#ef4444",
        "cancelled": "#6b7280"
    }
    status_color = status_colors.get(status, "#6b7280")

    # Priority badge
    priority_badges = {
        "high": '<span class="priority-badge high">⚡ High</span>',
        "normal": '<span class="priority-badge normal">Normal</span>',
        "low": '<span class="priority-badge low">Low</span>'
    }
    priority_badge = priority_badges.get(priority, "")

    # Progress bar (only for running jobs)
    progress_html = ""
    if status == "running":
        progress_html = f"""
        <div class="job-progress">
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress:.1f}%;"></div>
            </div>
            <span class="progress-text">{progress:.1f}% - {progress_message}</span>
        </div>
        """

    # Action buttons
    actions_html = ""
    if show_actions:
        if status in ["pending", "queued", "running"]:
            actions_html = f"""
            <div class="job-actions">
                <button class="action-btn cancel-btn" onclick="cancelJob('{job_id}')">
                    ⛔ Cancel
                </button>
            </div>
            """
        elif status == "completed":
            actions_html = f"""
            <div class="job-actions">
                <button class="action-btn view-btn" onclick="viewJob('{job_id}')">
                    👁️ View
                </button>
                <button class="action-btn download-btn" onclick="downloadJob('{job_id}')">
                    ⬇️ Download
                </button>
            </div>
            """

    return f"""
    <div class="job-card" data-job-id="{job_id}">
        <div class="job-header">
            <span class="job-id">#{job_id[:8]}</span>
            {priority_badge}
            <span class="job-status" style="color: {status_color};">
                ● {status.upper()}
            </span>
        </div>
        <div class="job-prompt">{prompt}</div>
        {progress_html}
        <div class="job-footer">
            <span class="job-time">{created_at}</span>
            {actions_html}
        </div>
    </div>
    """


def render_job_list(jobs: List[Dict[str, Any]], empty_message: str = "No jobs") -> str:
    """
    Render a list of job cards.

    Args:
        jobs: List of job data dictionaries
        empty_message: Message to show when list is empty
    """
    if not jobs:
        return render_empty_state(empty_message)

    cards = [render_job_card(job) for job in jobs]
    return f"""
    <div class="job-list">
        {"".join(cards)}
    </div>
    """


def format_job_for_display(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format job data for display.

    Converts datetime strings to readable format and truncates long text.
    """
    formatted = job_data.copy()

    # Format datetime
    if formatted.get("created_at"):
        try:
            dt = datetime.fromisoformat(formatted["created_at"].replace("Z", "+00:00"))
            formatted["created_at"] = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            pass

    # Truncate prompt
    if len(formatted.get("prompt", "")) > 150:
        formatted["prompt"] = formatted["prompt"][:150] + "..."

    return formatted

