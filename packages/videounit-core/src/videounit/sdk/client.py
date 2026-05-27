"""VideoUnit HTTP client for backend communication."""

import logging
from pathlib import Path
from typing import Optional

import httpx

from videounit.core.config import VideoUnitConfig
from videounit.core.errors import ConnectionError, TimeoutError, EvaluationError
from videounit.sdk.models import VideoContract, EvaluationResult

log = logging.getLogger(__name__)


class VideoUnitClient:
    """Async HTTP client for VideoUnit backend.

    Provides methods to:
    - Evaluate videos against contracts
    - Generate contracts from prompts
    - Get evaluation reports
    - Check backend health
    """

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """Initialize VideoUnit client.

        Args:
            backend_url: URL of the VideoUnit backend server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
        """
        self.config = VideoUnitConfig(
            backend_url=backend_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def backend_url(self) -> str:
        return self.config.backend_url

    @property
    def api_key(self) -> Optional[str]:
        return self.config.api_key

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "VideoUnitClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def health_check(self) -> bool:
        """Check if the backend is available.

        Returns:
            True if backend is healthy, False otherwise.
        """
        try:
            resp = await self.client.get("/api/health")
            return resp.status_code == 200
        except Exception as e:
            log.debug(f"Health check failed: {e}")
            return False

    async def evaluate(
        self, video_path: str, contract: VideoContract
    ) -> EvaluationResult:
        """Run full evaluation on a video against a contract.

        Args:
            video_path: Path to the video file.
            contract: VideoContract specifying test requirements.

        Returns:
            EvaluationResult with pass/fail status and evidence.

        Raises:
            ConnectionError: If cannot connect to backend.
            TimeoutError: If request times out.
            EvaluationError: If evaluation fails.
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise EvaluationError(f"Video file not found: {video_path}")

        try:
            with open(video_path, "rb") as f:
                files = {"video": (video_path_obj.name, f, "video/mp4")}
                data = {
                    "contract": contract.model_dump_json(),
                }
                resp = await self.client.post(
                    "/api/videounit/evaluate",
                    files=files,
                    data=data,
                )

            if resp.status_code == 200:
                result_data = resp.json()
                return EvaluationResult(**result_data)
            elif resp.status_code == 401:
                raise EvaluationError("Authentication failed. Check your API key.")
            else:
                error_msg = resp.text or f"HTTP {resp.status_code}"
                raise EvaluationError(f"Evaluation failed: {error_msg}")

        except httpx.TimeoutException:
            raise TimeoutError(f"Evaluation timed out after {self.config.timeout}s")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {e}")

    async def generate_contract(self, prompt: str) -> VideoContract:
        """Generate a contract from a text prompt using VLM.

        Args:
            prompt: Text description of the video test requirements.

        Returns:
            Generated VideoContract.

        Raises:
            ConnectionError: If cannot connect to backend.
            TimeoutError: If request times out.
            EvaluationError: If contract generation fails.
        """
        try:
            resp = await self.client.post(
                "/api/videounit/contract/generate",
                json={"prompt": prompt},
            )

            if resp.status_code == 200:
                contract_data = resp.json()
                return VideoContract(**contract_data)
            elif resp.status_code == 401:
                raise EvaluationError("Authentication failed. Check your API key.")
            else:
                error_msg = resp.text or f"HTTP {resp.status_code}"
                raise EvaluationError(f"Contract generation failed: {error_msg}")

        except httpx.TimeoutException:
            raise TimeoutError(f"Contract generation timed out after {self.config.timeout}s")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(f"Contract generation failed: {e}")

    async def get_report(self, run_id: str) -> str:
        """Get HTML report for a evaluation run.

        Args:
            run_id: The evaluation run ID.

        Returns:
            HTML report content.

        Raises:
            ConnectionError: If cannot connect to backend.
            TimeoutError: If request times out.
            EvaluationError: If report retrieval fails.
        """
        try:
            resp = await self.client.get(f"/api/videounit/report/{run_id}")

            if resp.status_code == 200:
                return resp.text
            elif resp.status_code == 404:
                raise EvaluationError(f"Report not found for run: {run_id}")
            elif resp.status_code == 401:
                raise EvaluationError("Authentication failed. Check your API key.")
            else:
                error_msg = resp.text or f"HTTP {resp.status_code}"
                raise EvaluationError(f"Report retrieval failed: {error_msg}")

        except httpx.TimeoutException:
            raise TimeoutError(f"Report retrieval timed out after {self.config.timeout}s")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(f"Report retrieval failed: {e}")

    async def get_result(self, run_id: str) -> EvaluationResult:
        """Get evaluation result for a completed run.

        Args:
            run_id: The evaluation run ID.

        Returns:
            EvaluationResult for the run.

        Raises:
            ConnectionError: If cannot connect to backend.
            TimeoutError: If request times out.
            EvaluationError: If result retrieval fails.
        """
        try:
            resp = await self.client.get(f"/api/videounit/result/{run_id}")

            if resp.status_code == 200:
                return EvaluationResult(**resp.json())
            elif resp.status_code == 404:
                raise EvaluationError(f"Result not found for run: {run_id}")
            elif resp.status_code == 401:
                raise EvaluationError("Authentication failed. Check your API key.")
            else:
                error_msg = resp.text or f"HTTP {resp.status_code}"
                raise EvaluationError(f"Result retrieval failed: {error_msg}")

        except httpx.TimeoutException:
            raise TimeoutError(f"Result retrieval timed out after {self.config.timeout}s")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to backend: {e}")
        except EvaluationError:
            raise
        except Exception as e:
            raise EvaluationError(f"Result retrieval failed: {e}")
