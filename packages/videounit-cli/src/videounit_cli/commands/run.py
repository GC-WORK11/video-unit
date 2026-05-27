"""Run VideoUnit evaluation on a video."""

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import typer

from videounit_cli.utils.backend import BackendClient, check_backend_health, get_backend_url
from videounit_cli.utils.config import load_config
from videounit_cli.utils.output import (
    console,
    print_error,
    print_success,
    print_warning,
    print_result,
    print_info,
    create_progress,
)


def run(
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
    """Run VideoUnit evaluation on a video against a contract.

    This command:
    1. Checks backend health
    2. Loads and validates the contract YAML
    3. Runs evaluation via the backend
    4. Prints summary results to console
    5. Generates report files

    Example:
        videounit run output.mp4 --contract tests/red_ball.yaml
        videounit run https://example.com/video.mp4 -c contract.yaml -b http://localhost:8000
    """
    backend_url = backend or get_backend_url()

    console.print(f"[cyan]VideoUnit Run[/cyan]")
    console.print(f"  Video: {video}")
    console.print(f"  Contract: {contract}")
    console.print(f"  Backend: {backend_url}")
    console.print(f"  Output: {output}")

    contract_path = Path(contract)
    if not contract_path.exists():
        print_error(f"Contract file not found: {contract}")
        raise typer.Exit(1)

    try:
        import yaml
        with open(contract_path, "r") as f:
            contract_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print_error(f"Invalid YAML in contract file: {e}")
        raise typer.Exit(1)

    console.print()
    console.print("[cyan]Checking backend health...[/cyan]")

    async def run_evaluation_with_polling():
        client = BackendClient(base_url=backend_url)

        try:
            health_ok = await check_backend_health(backend_url)
            if not health_ok:
                print_error("Backend is not healthy. Cannot proceed with evaluation.")
                raise typer.Exit(1)

            print_success("Backend is healthy")

            console.print()
            console.print("[cyan]Starting evaluation...[/cyan]")

            result = await client.evaluate(
                video_path=video,
                contract=contract_data,
                output_dir=output,
            )

            # If status is running, poll for completion
            if result.get("status") == "running":
                run_id = result.get("run_id")
                console.print(f"[cyan]Waiting for evaluation to complete...[/cyan] (run_id: {run_id})")

                max_wait = 300  # 5 minutes max
                waited = 0
                while waited < max_wait:
                    await asyncio.sleep(5)
                    run_result = await client.get_run(run_id)
                    if run_result.get("status") == "completed":
                        result = run_result.get("result", run_result)
                        break
                    elif run_result.get("status") == "failed":
                        print_error(f"Evaluation failed: {run_result.get('error', 'Unknown error')}")
                        raise typer.Exit(1)
                    waited += 5
                    console.print(f"[dim]Still running... ({waited}s elapsed)[/dim]")
                else:
                    print_error("Evaluation timed out after 5 minutes")
                    raise typer.Exit(1)

            return result

        finally:
            await client.close()

    result = asyncio.run(run_evaluation_with_polling())

    print_result(result)

    run_id = result.get("run_id", "unknown")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_json_path = run_dir / "run.json"
    with open(run_json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print_success(f"Results saved: {run_json_path}")

    if format in ("html", "both"):
        report_path = run_dir / "report.html"
        console.print(f"[dim]Report would be generated at: {report_path}[/dim]")

    if format in ("json", "both"):
        console.print(f"[dim]JSON results saved at: {run_json_path}[/dim]")

    console.print()
    print_success(f"Run completed: {run_id}")
