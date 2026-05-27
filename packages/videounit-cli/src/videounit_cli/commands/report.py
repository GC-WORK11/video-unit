"""Generate reports from VideoUnit runs."""

import json
from pathlib import Path

import httpx
import typer

from videounit_cli.utils.backend import get_backend_url
from videounit_cli.utils.output import console, print_error, print_success


app = typer.Typer(help="Generate HTML/JSON report from a previous run")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID or path to run directory"),
    format: str = typer.Option(
        "html",
        "--format",
        "-f",
        help="Report format: html, json, or both",
    ),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for report (default: runs/{run_id}/report.{format})",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Backend URL",
    ),
) -> None:
    """Generate HTML or JSON report from a previous run.

    Example:
        videounit report run_abc123
        videounit report run_abc123 --format json -o ./my_report.json
    """
    backend_url = backend or get_backend_url()

    # Check if it's a local path
    local_path = Path(run_id)
    if local_path.exists() and local_path.is_dir():
        # Local run directory
        _handle_local_report(local_path, format, output)
        return

    # Otherwise fetch from backend
    import asyncio
    asyncio.run(_handle_backend_report(run_id, format, output, backend_url))


def _handle_local_report(run_dir: Path, format: str, output: str | None):
    """Handle report from local run directory."""
    run_json = run_dir / "run.json"

    if not run_json.exists():
        print_error(f"Run result not found: {run_json}")
        raise typer.Exit(1)

    data = json.loads(run_json.read_text())

    if format in ("json", "both"):
        out_path = Path(output) if output else run_dir / "report.json"
        out_path.write_text(json.dumps(data, indent=2, default=str))
        print_success(f"JSON report saved: {out_path}")

    if format in ("html", "both"):
        html_path = run_dir / "report.html"
        if html_path.exists():
            out_path = Path(output) if output else html_path
            console.print(f"[dim]HTML report: {html_path}[/dim]")
        else:
            console.print("[yellow]HTML report not yet generated[/yellow]")
            console.print("[dim]Run evaluation first with --format both[/dim]")

    print_success(f"Report for run: {run_dir.name}")


async def _handle_backend_report(run_id: str, format: str, output: str | None, backend_url: str):
    """Handle report fetched from backend."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        url = f"{backend_url}/api/videounit/report/{run_id}"

        try:
            response = await client.get(url, params={"format": format})

            if response.status_code == 404:
                print_error(f"Run {run_id} not found")
                raise typer.Exit(1)

            response.raise_for_status()

            if format == "json":
                data = response.json()
                out_path = Path(output) if output else Path("runs") / run_id / "report.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(data, indent=2))
                print_success(f"JSON report saved: {out_path}")
            else:
                out_path = Path(output) if output else Path("runs") / run_id / "report.html"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(response.content)
                print_success(f"HTML report saved: {out_path}")

        except httpx.HTTPError as e:
            print_error(f"Failed to fetch report: {e}")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
