"""Core module for VideoUnit configuration and errors."""

from videounit.core.config import VideoUnitConfig
from videounit.core.errors import (
    VideoUnitError,
    EvaluationError,
    ContractError,
    ConnectionError,
    TimeoutError,
)

__all__ = [
    "VideoUnitConfig",
    "VideoUnitError",
    "EvaluationError",
    "ContractError",
    "ConnectionError",
    "TimeoutError",
]
