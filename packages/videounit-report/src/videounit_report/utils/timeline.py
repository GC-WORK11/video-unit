"""Timeline builder utilities."""

from datetime import datetime
from typing import Any


def build_failure_timeline(failures: list) -> list[dict]:
    """
    Convert failures to sorted timeline with color coding.

    Args:
        failures: List of failure dicts with frame_number, timestamp, severity, type, message

    Returns:
        Sorted list of timeline events with position and color coding
    """
    timeline = []

    severity_order = {
        "critical": 0,
        "fail": 1,
        "warning": 2,
        "info": 3,
    }

    for failure in failures:
        frame = failure.get("frame_number", 0)
        timestamp = failure.get("timestamp", "00:00:00")
        severity = failure.get("severity", "fail")
        ftype = failure.get("type", "assertion")
        message = failure.get("message", "No details")

        # Normalize severity
        severity_lower = severity.lower() if isinstance(severity, str) else "fail"

        timeline.append({
            "frame": frame,
            "timestamp": _format_timestamp(timestamp),
            "severity": severity_lower,
            "type": ftype,
            "message": message,
            "color": _get_severity_color(severity_lower),
            "position": 0,  # Calculated later based on video duration
        })

    # Sort by frame number
    timeline.sort(key=lambda x: x["frame"])

    # Assign sort order within same frame
    frame_groups = {}
    for event in timeline:
        frame = event["frame"]
        if frame not in frame_groups:
            frame_groups[frame] = 0
        else:
            frame_groups[frame] += 1
        event["order"] = frame_groups[frame]

    return timeline


def _format_timestamp(timestamp: Any) -> str:
    """Format timestamp for display in timeline."""
    if not timestamp:
        return "00:00:00"

    if isinstance(timestamp, (int, float)):
        # Assume frame number or seconds
        seconds = float(timestamp)
        if seconds > 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"

    if isinstance(timestamp, str):
        # Already formatted, just return
        return timestamp

    return str(timestamp)


def _get_severity_color(severity: str) -> str:
    """Return hex color code for severity level."""
    colors = {
        "critical": "#7c3aed",  # Purple
        "fail": "#ef4444",       # Red
        "warning": "#f59e0b",    # Amber
        "info": "#6b7280",       # Gray
    }
    return colors.get(severity.lower(), "#ef4444")


def build_timeline_markers(timeline: list[dict], video_duration: float) -> list[dict]:
    """
    Calculate timeline marker positions for video overlay.

    Args:
        timeline: List of timeline events
        video_duration: Total video duration in seconds

    Returns:
        Timeline events with position percentage calculated
    """
    markers = []
    fps = 30  # Assume 30fps

    for event in timeline:
        frame = event.get("frame", 0)
        # Calculate position as percentage of video
        time_seconds = frame / fps if fps > 0 else 0
        position = (time_seconds / video_duration * 100) if video_duration > 0 else 0
        position = min(position, 100)  # Cap at 100%

        markers.append({
            **event,
            "position": position,
            "time_seconds": time_seconds,
        })

    return markers


def group_timeline_by_severity(timeline: list[dict]) -> dict[str, list[dict]]:
    """
    Group timeline events by severity level.

    Args:
        timeline: List of timeline events

    Returns:
        Dict mapping severity -> list of events
    """
    grouped = {
        "critical": [],
        "fail": [],
        "warning": [],
        "info": [],
    }

    for event in timeline:
        severity = event.get("severity", "fail")
        if severity in grouped:
            grouped[severity].append(event)

    return grouped


def get_timeline_summary(timeline: list[dict]) -> dict[str, int]:
    """
    Get summary counts by severity.

    Args:
        timeline: List of timeline events

    Returns:
        Dict with count per severity level
    """
    summary = {
        "critical": 0,
        "fail": 0,
        "warning": 0,
        "info": 0,
        "total": len(timeline),
    }

    for event in timeline:
        severity = event.get("severity", "fail")
        if severity in summary:
            summary[severity] += 1

    return summary
