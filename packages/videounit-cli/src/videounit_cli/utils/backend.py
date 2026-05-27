"""Backend client for VideoUnit CLI.

Handles HTTP communication with the AETHER backend.
"""

import os
from pathlib import Path
from typing import Any

import httpx

from videounit_cli.utils.output import console, print_error, print_warning


DEFAULT_BACKEND_URL = "http://localhost:8000"


def get_backend_url() -> str:
    """Get the configured backend URL.

    Returns:
        Backend URL string, defaulting to localhost:8000.
    """
    return os.environ.get("VIDEOUNIT_BACKEND_URL", DEFAULT_BACKEND_URL)


class BackendClient:
    """HTTP client for communicating with the AETHER backend.

    Attributes:
        base_url: Base URL of the backend API.
        client: httpx AsyncClient for making requests.
    """

    def __init__(self, base_url: str | None = None):
        """Initialize the backend client.

        Args:
            base_url: Base URL for the backend API. Defaults to configured or localhost:8000.
        """
        self.base_url = base_url or get_backend_url()
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def health_check(self) -> dict[str, Any]:
        """Check backend health status.

        Returns:
            Health check response containing status, version, and uptime.

        Raises:
            httpx.HTTPError: If the health check fails.
        """
        response = await self.client.get("/api/health")
        response.raise_for_status()
        return response.json()

    async def upload_video(self, video_path: str, session_id: str | None = None) -> dict[str, Any]:
        """Upload a video file to the backend.

        Args:
            video_path: Path to the video file to upload.
            session_id: Optional session ID to associate with the upload.

        Returns:
            Video info response with video_id and metadata.

        Raises:
            FileNotFoundError: If the video file doesn't exist.
            httpx.HTTPError: If the upload fails.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        session_id = session_id or video_path.stem

        with open(video_path, "rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            response = await self.client.post(
                f"/api/videos/upload/{session_id}",
                files=files,
            )

        response.raise_for_status()
        return response.json()

    async def run_perception(
        self,
        session_id: str,
        max_frames: int = 16,
        run_3d: bool = False,
    ) -> dict[str, Any]:
        """Run perception pipeline on a session.

        Args:
            session_id: Session ID with uploaded video.
            max_frames: Maximum number of frames to process.
            run_3d: Whether to run 3D reconstruction.

        Returns:
            Perception results with segmentation, depth, and tracking data.

        Raises:
            httpx.HTTPError: If the perception run fails.
        """
        payload = {
            "max_frames": max_frames,
            "run_3d": run_3d,
        }
        response = await self.client.post(
            f"/api/perception/{session_id}/run",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def evaluate(
        self,
        video_path: str,
        contract: dict[str, Any],
        output_dir: str = "runs",
    ) -> dict[str, Any]:
        """Run VideoUnit evaluation on a video against a contract.

        This endpoint uploads the video, runs perception, and evaluates
        all assertions in the contract.

        Args:
            video_path: Path to the video file or URL.
            contract: Contract dictionary with test assertions.
            output_dir: Directory to save run results.

        Returns:
            Evaluation result with overall score, category scores, and failures.

        Raises:
            httpx.HTTPError: If the evaluation fails.
        """
        import uuid

        session_id = str(uuid.uuid4())[:8]
        video_path_obj = Path(video_path)

        if video_path.startswith("http://") or video_path.startswith("https://"):
            payload = {
                "video_url": video_path,
                "contract": contract,
                "output_dir": output_dir,
            }
            response = await self.client.post(
                "/api/videounit/evaluate",
                json=payload,
                params={"session_id": session_id},
            )
        elif video_path_obj.exists():
            with open(video_path_obj, "rb") as f:
                files = {"video": (video_path_obj.name, f, "video/mp4")}
                data = {
                    "contract_yaml": yaml_safe_dump(contract),
                    "output_dir": output_dir,
                }
                response = await self.client.post(
                    "/api/videounit/evaluate",
                    files=files,
                    data=data,
                    params={"session_id": session_id},
                )
        else:
            raise FileNotFoundError(f"Video file not found: {video_path}")

        response.raise_for_status()
        return response.json()

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """Get the status and results of a previous run.

        Args:
            run_id: The run ID to retrieve.

        Returns:
            Run results including scores and failures.

        Raises:
            httpx.HTTPError: If the run is not found or retrieval fails.
        """
        response = await self.client.get(f"/api/videounit/run/{run_id}")
        response.raise_for_status()
        return response.json()

    async def generate_report(
        self,
        run_id: str,
        format: str = "html",
    ) -> dict[str, Any]:
        """Generate a report from a run.

        Args:
            run_id: The run ID to generate a report from.
            format: Report format ('html', 'json', or 'both').

        Returns:
            Report metadata including paths to generated files.

        Raises:
            httpx.HTTPError: If report generation fails.
        """
        response = await self.client.get(
            f"/api/videounit/report/{run_id}",
            params={"format": format},
        )
        response.raise_for_status()
        return response.json()

    async def generate_contract(
        self,
        prompt: str,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        """Generate a test contract from a text prompt using VLM.

        Args:
            prompt: Text description of the video to test.
            output_path: Optional path to save the generated contract YAML.

        Returns:
            Generated contract dictionary.

        Raises:
            httpx.HTTPError: If contract generation fails.
        """
        payload = {"prompt": prompt}
        response = await self.client.post(
            "/api/videounit/contract/generate",
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        if output_path and "contract" in result:
            contract_yaml = result["contract"]
            if isinstance(contract_yaml, dict):
                import yaml
                contract_yaml = yaml.dump(contract_yaml, default_flow_style=False)

            with open(output_path, "w") as f:
                f.write(contract_yaml)

            result["output_path"] = output_path

        return result


def yaml_safe_dump(data: dict[str, Any]) -> str:
    """Safely dump data to YAML string.

    Args:
        data: Dictionary to convert to YAML.

    Returns:
        YAML string representation.
    """
    import yaml
    return yaml.dump(data, default_flow_style=False)


async def check_backend_health(base_url: str | None = None) -> bool:
    """Check if the backend is healthy and accessible.

    Args:
        base_url: Optional backend URL to check.

    Returns:
        True if backend is healthy, False otherwise.
    """
    url = base_url or get_backend_url()
    client = BackendClient(base_url=url)

    try:
        health = await client.health_check()
        status = health.get("status", "unknown")
        version = health.get("version", "unknown")
        console.print(f"[dim]Backend {url} - status: {status}, version: {version}[/dim]")
        await client.close()
        return status == "healthy"
    except httpx.ConnectError:
        print_error(f"Cannot connect to backend at {url}")
        print_warning("Make sure the backend is running with: videounit serve")
        await client.close()
        return False
    except httpx.HTTPError as e:
        print_error(f"Backend error: {e}")
        await client.close()
        return False
