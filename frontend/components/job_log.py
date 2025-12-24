"""
Job Log Component
=================
Real-time log viewer for generation progress and system events.
"""

import gradio as gr
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class LogLevel(str, Enum):
    """Log level types."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    job_id: Optional[str] = None

    def to_html(self) -> str:
        """Convert to HTML representation."""
        level_icons = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.SUCCESS: "✅"
        }

        level_classes = {
            LogLevel.DEBUG: "log-debug",
            LogLevel.INFO: "log-info",
            LogLevel.WARNING: "log-warning",
            LogLevel.ERROR: "log-error",
            LogLevel.SUCCESS: "log-success"
        }

        icon = level_icons.get(self.level, "•")
        css_class = level_classes.get(self.level, "log-info")
        time_str = self.timestamp.strftime("%H:%M:%S")
        job_tag = f'<span class="log-job-id">[{self.job_id[:8]}]</span>' if self.job_id else ""

        return f"""
        <div class="log-entry {css_class}">
            <span class="log-time">{time_str}</span>
            <span class="log-icon">{icon}</span>
            {job_tag}
            <span class="log-message">{self.message}</span>
        </div>
        """


class LogManager:
    """Manages log entries with a fixed-size buffer."""

    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self.entries: List[LogEntry] = []

    def add(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        job_id: Optional[str] = None
    ):
        """Add a new log entry."""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            job_id=job_id
        )
        self.entries.append(entry)

        # Trim to max size
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def debug(self, message: str, job_id: Optional[str] = None):
        """Add a debug log."""
        self.add(message, LogLevel.DEBUG, job_id)

    def info(self, message: str, job_id: Optional[str] = None):
        """Add an info log."""
        self.add(message, LogLevel.INFO, job_id)

    def warning(self, message: str, job_id: Optional[str] = None):
        """Add a warning log."""
        self.add(message, LogLevel.WARNING, job_id)

    def error(self, message: str, job_id: Optional[str] = None):
        """Add an error log."""
        self.add(message, LogLevel.ERROR, job_id)

    def success(self, message: str, job_id: Optional[str] = None):
        """Add a success log."""
        self.add(message, LogLevel.SUCCESS, job_id)

    def clear(self):
        """Clear all log entries."""
        self.entries = []

    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        job_id: Optional[str] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Get filtered log entries."""
        filtered = self.entries

        if level:
            filtered = [e for e in filtered if e.level == level]

        if job_id:
            filtered = [e for e in filtered if e.job_id == job_id]

        return filtered[-limit:]

    def render_html(
        self,
        level: Optional[LogLevel] = None,
        job_id: Optional[str] = None,
        limit: int = 100
    ) -> str:
        """Render logs as HTML."""
        entries = self.get_entries(level, job_id, limit)

        if not entries:
            return """
            <div class="log-empty">
                <span class="empty-icon">📋</span>
                <span class="empty-text">No log entries</span>
            </div>
            """

        html_entries = [e.to_html() for e in reversed(entries)]
        return f"""
        <div class="log-container">
            {"".join(html_entries)}
        </div>
        """


# Global log manager instance
log_manager = LogManager()


def create_job_log() -> dict:
    """
    Create the job log component.

    Returns:
        Dictionary of log components
    """
    with gr.Column(elem_classes=["job-log-panel"]) as panel:
        # Header
        with gr.Row():
            gr.Markdown(
                """
                ## 📝 Activity Log
                """,
                elem_classes=["panel-header"]
            )

            with gr.Row():
                level_filter = gr.Dropdown(
                    choices=[
                        ("All Levels", "all"),
                        ("Debug", "debug"),
                        ("Info", "info"),
                        ("Warning", "warning"),
                        ("Error", "error"),
                        ("Success", "success")
                    ],
                    value="all",
                    label="Filter",
                    scale=1,
                    interactive=True
                )

                clear_log_btn = gr.Button(
                    "🗑️ Clear",
                    size="sm",
                    variant="secondary"
                )

        # Log Display
        log_display = gr.HTML(
            render_log_empty(),
            elem_classes=["log-display"]
        )

        # Auto-scroll toggle
        auto_scroll = gr.Checkbox(
            label="Auto-scroll to latest",
            value=True,
            elem_classes=["auto-scroll-toggle"]
        )

    return {
        "panel": panel,
        "log_display": log_display,
        "level_filter": level_filter,
        "clear_log_btn": clear_log_btn,
        "auto_scroll": auto_scroll
    }


def render_log_empty() -> str:
    """Render empty log state."""
    return """
    <div class="log-empty">
        <span class="empty-icon">📋</span>
        <span class="empty-text">Waiting for activity...</span>
    </div>
    """


def render_log_entries(entries: List[dict]) -> str:
    """
    Render log entries from dictionaries.

    Args:
        entries: List of log entry dictionaries with keys:
                 - timestamp (str or datetime)
                 - level (str)
                 - message (str)
                 - job_id (optional str)
    """
    if not entries:
        return render_log_empty()

    html_parts = []

    for entry in entries:
        level = entry.get("level", "info")
        message = entry.get("message", "")
        job_id = entry.get("job_id")
        timestamp = entry.get("timestamp")

        # Parse timestamp
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                timestamp = datetime.now()
        elif timestamp is None:
            timestamp = datetime.now()

        log_entry = LogEntry(
            timestamp=timestamp,
            level=LogLevel(level) if level in [l.value for l in LogLevel] else LogLevel.INFO,
            message=message,
            job_id=job_id
        )
        html_parts.append(log_entry.to_html())

    return f"""
    <div class="log-container" id="log-container">
        {"".join(reversed(html_parts))}
    </div>
    """


def add_log_entry(
    message: str,
    level: str = "info",
    job_id: Optional[str] = None
) -> str:
    """
    Add a log entry and return updated HTML.

    Args:
        message: Log message
        level: Log level (debug, info, warning, error, success)
        job_id: Optional job ID to associate with

    Returns:
        Updated log HTML
    """
    log_manager.add(
        message=message,
        level=LogLevel(level) if level in [l.value for l in LogLevel] else LogLevel.INFO,
        job_id=job_id
    )
    return log_manager.render_html()


def clear_logs() -> str:
    """Clear all logs and return empty HTML."""
    log_manager.clear()
    return render_log_empty()


def filter_logs(level: str) -> str:
    """
    Filter logs by level.

    Args:
        level: Level to filter by, or "all" for no filter

    Returns:
        Filtered log HTML
    """
    if level == "all":
        return log_manager.render_html()

    try:
        level_enum = LogLevel(level)
        return log_manager.render_html(level=level_enum)
    except ValueError:
        return log_manager.render_html()

