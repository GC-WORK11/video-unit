"""Evidence frame extraction utilities."""

from pathlib import Path
from typing import Any


def extract_evidence_frames(
    video_path: str,
    failures: list,
    output_dir: Path,
    max_frames: int = 20,
) -> list[dict]:
    """
    Extract thumbnail images for failure evidence using OpenCV.

    Args:
        video_path: Path to the video file
        failures: List of failure dicts with frame_number and timestamp
        output_dir: Directory to write frame images
        max_frames: Maximum number of frames to extract

    Returns:
        List of dicts with frame metadata including path, frame_number, timestamp, severity, message
    """
    try:
        import cv2
    except ImportError:
        # OpenCV not available, return empty evidence list
        return _extract_frames_pil(video_path, failures, output_dir, max_frames)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_frames = []

    if not video_path or not Path(video_path).exists():
        return _build_evidence_from_failures(failures, output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _build_evidence_from_failures(failures, output_dir)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Deduplicate and limit frames
    seen_frames = set()
    for failure in failures:
        frame_num = failure.get("frame_number")
        if frame_num is None:
            continue

        if frame_num in seen_frames:
            continue
        if len(evidence_frames) >= max_frames:
            break

        seen_frames.add(frame_num)

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            # Generate output filename
            timestamp = failure.get("timestamp", "00:00:00")
            severity = failure.get("severity", "fail")
            output_path = output_dir / f"frame_{frame_num:06d}_{severity}.jpg"

            # Save frame with quality compression
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            evidence_frames.append({
                "frame_number": frame_num,
                "timestamp": timestamp,
                "path": str(output_path),
                "severity": severity,
                "type": failure.get("type", "assertion"),
                "message": failure.get("message", ""),
            })

    cap.release()

    # Sort by frame number
    evidence_frames.sort(key=lambda x: x["frame_number"])

    return evidence_frames


def _extract_frames_pil(
    video_path: str,
    failures: list,
    output_dir: Path,
    max_frames: int = 20,
) -> list[dict]:
    """Fallback frame extraction using PIL when OpenCV unavailable."""
    # If we can't extract frames, just build evidence from failure data
    return _build_evidence_from_failures(failures, output_dir)


def _build_evidence_from_failures(failures: list, output_dir: Path) -> list[dict]:
    """Build evidence list from failures without actual frame extraction."""
    evidence = []
    for failure in failures:
        frame_num = failure.get("frame_number")
        if frame_num is None:
            continue

        evidence.append({
            "frame_number": frame_num,
            "timestamp": failure.get("timestamp", "00:00:00"),
            "path": "",
            "severity": failure.get("severity", "fail"),
            "type": failure.get("type", "assertion"),
            "message": failure.get("message", ""),
        })

    return evidence[:20]
