"""Start the VideoUnit backend server."""

import os
import subprocess
import sys

import typer

from videounit_cli.utils.output import console, print_error, print_success


app = typer.Typer(help="Start the VideoUnit backend server")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the server on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind the server to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload on code changes"),
    backend_path: str = typer.Option(
        None,
        "--backend-path",
        help="Path to backend directory (default: looking for backend/ in project root)",
    ),
) -> None:
    """Start uvicorn backend server.

    Example:
        videounit serve
        videounit serve --port 9000 --host 0.0.0.0
        videounit serve --reload
    """
    # Find backend path
    if backend_path is None:
        from pathlib import Path

        cli_path = Path(__file__).resolve().parent
        # Go up from commands/ to videounit_cli/ to src/ to videounit-cli/ to packages/ to video-unit/
        project_root = cli_path.parent.parent.parent.parent.parent
        potential_backend = project_root / "backend"

        if potential_backend.exists():
            backend_path = str(potential_backend)
        else:
            # Check current working directory
            cwd_backend = Path.cwd() / "backend"
            if cwd_backend.exists():
                backend_path = str(cwd_backend)
            else:
                print_error("Could not find backend directory. Specify --backend-path or run from project root.")
                raise typer.Exit(1)

    backend_app_path = Path(backend_path) / "app" / "main.py"
    if not backend_app_path.exists():
        print_error(f"Backend app not found at: {backend_app_path}")
        print_error("Ensure the backend structure is: <backend>/app/main.py")
        raise typer.Exit(1)

    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port),
    ]

    if reload:
        cmd.append("--reload")

    console.print(f"[cyan]Starting VideoUnit backend server...[/cyan]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Backend: {backend_path}")
    if reload:
        console.print(f"  [yellow]Reload: enabled[/yellow]")
    console.print()
    console.print("[dim]Press Ctrl+C to stop[/dim]")
    console.print()

    # Build environment with correct PYTHONPATH
    # Put video-unit backend FIRST so it takes priority over any other Python paths
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")

    # video-unit backend goes first, then preserve rest of PYTHONPATH
    if current_pythonpath:
        new_pythonpath = f"{backend_path}:{current_pythonpath}"
    else:
        new_pythonpath = backend_path

    env["PYTHONPATH"] = new_pythonpath

    try:
        # Run from backend directory with corrected PYTHONPATH
        subprocess.run(cmd, cwd=backend_path, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print_error(f"Server exited with error: {e}")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print()
        print_success("Server stopped")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
