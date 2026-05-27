"""Videos API - upload and extract frames."""
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from app.core.config import DATA_DIR
from app.core.errors import session_not_found

router = APIRouter(prefix="/api/videos")
_videos: dict = {}


class VideoInfo(BaseModel):
    video_id: str
    session_id: str
    filename: str
    size_bytes: int
    duration_seconds: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    status: str


@router.post("/upload/{session_id}", response_model=VideoInfo)
async def upload_video(session_id: str, file: UploadFile = File(...)):
    from app.api.sessions import _sessions
    if session_id not in _sessions:
        raise session_not_found(session_id)

    session_dir = DATA_DIR / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    video_id = str(uuid.uuid4())[:8]
    video_path = session_dir / "video.mp4"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    size = video_path.stat().st_size
    metadata = _probe_video(video_path)

    video_data = {
        "video_id": video_id,
        "session_id": session_id,
        "filename": file.filename,
        "size_bytes": size,
        "duration_seconds": metadata.get("duration"),
        "fps": metadata.get("fps"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "status": "uploaded",
        "path": str(video_path),
    }
    _videos[video_id] = video_data
    _sessions[session_id]["video_loaded"] = True
    _sessions[session_id]["status"] = "video_loaded"

    return VideoInfo(**{k: v for k, v in video_data.items() if k != "path"})


@router.get("/videos/{video_id}", response_model=VideoInfo)
async def get_video(video_id: str):
    if video_id not in _videos:
        raise HTTPException(status_code=404, detail=f"Video not found: {video_id}")
    return VideoInfo(**{k: v for k, v in _videos[video_id].items() if k != "path"})


def _probe_video(path: Path) -> dict:
    try:
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            return {
                "fps": float(stream.average_rate),
                "width": stream.width,
                "height": stream.height,
                "duration": float(container.duration / 1_000_000),
            }
    except Exception:
        return {}


from fastapi import HTTPException
