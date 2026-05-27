"""Evaluate command - alias for run command."""

import typer

from videounit_cli.commands.run import run as run_command


def evaluate(
    video: str = typer.Argument(..., help="Path to video file or URL"),
    contract: str = typer.Option(
        ...,
        "--contract",
        "-c",
        help="Path to contract YAML file",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Backend URL (default: from config or http://localhost:8000)",
    ),
    output: str = typer.Option(
        "runs/",
        "--output",
        "-o",
        help="Output directory for run results",
    ),
    format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Report format: html, json, or both",
    ),
) -> None:
    """Evaluate a video against a contract (alias for 'run' command).

    This is an alias for 'videounit run'. See 'videounit run --help' for details.

    Example:
        videounit evaluate output.mp4 --contract tests/red_ball.yaml
    """
    run_command(
        video=video,
        contract=contract,
        backend=backend,
        output=output,
        format=format,
    )
