"""Sessions API."""
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import DATA_DIR
from app.core.errors import session_not_found

router = APIRouter(prefix="/api/sessions")

_sessions: dict = {}


class SessionInfo(BaseModel):
    id: str
    name: str
    created_at: str
    status: str
    video_loaded: bool = False
    frame_count: int = 0
    has_scene_graph: bool = False


@router.post("", response_model=SessionInfo)
async def create_session(name: str | None = None):
    session_id = str(uuid.uuid4())[:8]
    session_dir = DATA_DIR / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "frames").mkdir(exist_ok=True)

    session = {
        "id": session_id,
        "name": name or f"Session {session_id}",
        "created_at": datetime.now().isoformat(),
        "status": "created",
        "video_loaded": False,
        "frame_count": 0,
        "has_scene_graph": False,
    }
    _sessions[session_id] = session
    return SessionInfo(**session)



@router.get("")
async def list_sessions():
    return [SessionInfo(**s) for s in _sessions.values()]


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise session_not_found(session_id)
    return SessionInfo(**_sessions[session_id])


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    import shutil
    if session_id not in _sessions:
        raise session_not_found(session_id)
    (DATA_DIR / "sessions" / session_id).exists() and shutil.rmtree(DATA_DIR / "sessions" / session_id)
    del _sessions[session_id]
    return {"deleted": session_id}


@router.patch("/{session_id}")
async def update_session(session_id: str, updates: dict):
    if session_id not in _sessions:
        raise session_not_found(session_id)
    _sessions[session_id].update(updates)
    return SessionInfo(**_sessions[session_id])
