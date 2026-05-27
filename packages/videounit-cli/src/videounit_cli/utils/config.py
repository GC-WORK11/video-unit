"""Configuration management for VideoUnit CLI."""

import os
import json
from pathlib import Path
from typing import Any

import yaml


def get_default_config_dir() -> Path:
    """Get the default configuration directory."""
    config_dir = Path.home() / ".config" / "videounit"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    return get_default_config_dir() / "config.yaml"


def load_config() -> dict[str, Any]:
    """Load CLI configuration from the default config file.

    Returns:
        Configuration dictionary with defaults for any missing values.
    """
    config_path = get_default_config_path()

    defaults: dict[str, Any] = {
        "backend_url": "http://localhost:8000",
        "default_output_dir": "runs",
        "default_format": "html",
        "project_config_file": "videounit.yaml",
    }

    if not config_path.exists():
        return defaults

    try:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        return {**defaults, **user_config}
    except Exception:
        return defaults


def save_config(config: dict[str, Any]) -> None:
    """Save CLI configuration to the default config file.

    Args:
        config: Configuration dictionary to save.
    """
    config_path = get_default_config_path()
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def load_project_config(project_dir: Path | str = ".") -> dict[str, Any] | None:
    """Load VideoUnit project configuration.

    Args:
        project_dir: Path to the project directory.

    Returns:
        Project configuration dictionary or None if not found.
    """
    project_dir = Path(project_dir)
    config_file = project_dir / "videounit.yaml"

    if not config_file.exists():
        return None

    try:
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def save_project_config(config: dict[str, Any], project_dir: Path | str = ".") -> None:
    """Save VideoUnit project configuration.

    Args:
        config: Configuration dictionary to save.
        project_dir: Path to the project directory.
    """
    project_dir = Path(project_dir)
    config_file = project_dir / "videounit.yaml"

    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
