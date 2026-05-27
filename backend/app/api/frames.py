"""Frames API - extract and manage video frames."""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import DATA_DIR, settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/frames")


class ExtractFramesRequest(BaseModel):
    session_id: str
    fps: float = 5.0
    max_frames: int | None = None
    start_time: float = 0.0
    end_time: float | None = None


class ExtractFramesResponse(BaseModel):
    session_id: str
    frame_count: int
    frames: list[dict]
    fps: float
    duration_seconds: float


@router.post("/extract", response_model=ExtractFramesResponse)
async def extract_frames(req: ExtractFramesRequest) -> ExtractFramesResponse:
    """Extract frames from a session's video."""
    from app.api.sessions import _sessions
    from app.api.videos import _videos

    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {req.session_id}")

    session = _sessions[req.session_id]
    # Find the actual video file in the session directory
    session_dir = DATA_DIR / "sessions" / req.session_id
    video_files = list(session_dir.glob("*.mp4")) + list(session_dir.glob("*.mov")) + list(session_dir.glob("*.avi")) + list(session_dir.glob("*.mkv"))
    if not video_files:
        raise HTTPException(status_code=404, detail="No video found for this session. Upload a video first.")
    video_path = video_files[0]  # Use first video found

    frames_dir = DATA_DIR / "sessions" / req.session_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Import the extractor
    from app.video.loader import FrameExtractor

    extractor = FrameExtractor(str(video_path))
    max_f = req.max_frames or settings.max_frames
    every_nth = max(1, int(extractor.metadata.frame_count / max_f)) if extractor.metadata.frame_count > max_f else 1

    extracted = extractor.extract_all_frames(frames_dir, every_nth=every_nth, max_frames=max_f)

    # Get frame info
    frames_info = []
    for i, frame_path in enumerate(extracted):
        frames_info.append({
            "frame_id": f"frame_{i:05d}",
            "index": i,
            "path": f"/static/{req.session_id}/frames/{frame_path.name}",
            "size_bytes": frame_path.stat().st_size,
            "timestamp": round(i * every_nth / extractor.metadata.fps, 3) if extractor.metadata.fps else 0,
        })

    # Update session
    _sessions[req.session_id]["frame_count"] = len(frames_info)
    _sessions[req.session_id]["status"] = "frames_extracted"

    return ExtractFramesResponse(
        session_id=req.session_id,
        frame_count=len(frames_info),
        frames=frames_info,
        fps=extractor.metadata.fps,
        duration_seconds=extractor.metadata.duration,
    )


@router.get("/{session_id}")
async def list_frames(session_id: str):
    """List all extracted frames for a session."""
    # Frames are stored in the frames subdirectory
    frames_dir = DATA_DIR / "sessions" / session_id / "frames"

    if not frames_dir.exists():
        return {"frames": [], "count": 0}

    frames = sorted(frames_dir.glob("frame_*.png"))
    return {
        "frames": [
            {
                "frame_id": f.stem,
                "index": int(f.stem.split("_")[1]),
                "path": f"/static/{session_id}/frames/{f.name}",
                "size_bytes": f.stat().st_size,
            }
            for f in frames
        ],
        "count": len(frames),
    }


@router.get("/{session_id}/{frame_id}")
async def get_frame(session_id: str, frame_id: str):
    """Get a specific frame."""
    # Frames are stored in the frames subdirectory
    frame_path = DATA_DIR / "sessions" / session_id / "frames" / f"{frame_id}.png"

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame not found: {frame_id}")

    return {
        "frame_id": frame_id,
        "path": f"/static/{session_id}/frames/{frame_path.name}",
        "size_bytes": frame_path.stat().st_size,
    }
