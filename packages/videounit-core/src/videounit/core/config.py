"""VideoUnit configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoUnitConfig:
    """Configuration for VideoUnit client.

    Attributes:
        backend_url: URL of the VideoUnit backend server.
        api_key: API key for authentication (optional for local development).
        timeout: Request timeout in seconds.
        max_frames: Maximum number of frames to extract for evaluation.
        checkpoint_dir: Directory for model checkpoints.
    """

    backend_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: float = 300.0
    max_frames: int = 3000
    checkpoint_dir: str = "/home/govinda/aether/data/checkpoints"

    def __post_init__(self):
        self.backend_url = self.backend_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "VideoUnitConfig":
        """Create config from environment variables."""
        import os
        return cls(
            backend_url=os.getenv("VIDEOUNIT_BACKEND_URL", "http://localhost:8000"),
            api_key=os.getenv("VIDEOUNIT_API_KEY"),
            timeout=float(os.getenv("VIDEOUNIT_TIMEOUT", "300.0")),
            max_frames=int(os.getenv("VIDEOUNIT_MAX_FRAMES", "3000")),
            checkpoint_dir=os.getenv(
                "VIDEOUNIT_CHECKPOINT_DIR",
                "/home/govinda/aether/data/checkpoints"
            ),
        )
