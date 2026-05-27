"""Configuration for AETHER Studio backend."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path(__file__).parent.parent.parent.parent / "data"
    models_dir: Path = Path(__file__).parent.parent.parent.parent / "models"
    host: str = "127.0.0.1"
    port: int = 8000
    default_fps: float = 5.0
    max_video_size_mb: int = 500
    max_frames: int = 10000
    simulation_horizon_default: float = 5.0
    simulation_timestep: float = 0.001
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None

    class Config:
        env_prefix = "AETHER_"


settings = Settings()
DATA_DIR = settings.data_dir
MODELS_DIR = settings.models_dir
