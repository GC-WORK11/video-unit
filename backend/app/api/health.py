"""Health check API."""
import platform
from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel
import psutil

from app import __version__

router = APIRouter(prefix="/api")
START_TIME = datetime.now()


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    memory_mb: dict


@router.get("/health", response_model=HealthResponse)
async def health_check():
    memory = psutil.virtual_memory()
    return HealthResponse(
        status="healthy",
        version=__version__,
        uptime_seconds=round((datetime.now() - START_TIME).total_seconds(), 1),
        memory_mb={
            "total": round(memory.total / 1024 ** 2, 0),
            "used": round(memory.used / 1024 ** 2, 0),
            "percent": memory.percent,
        },
    )
