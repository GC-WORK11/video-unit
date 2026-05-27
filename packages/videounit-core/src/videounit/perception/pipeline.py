"""Perception pipeline for VideoUnit - wraps AETHER backend perception."""

from dataclasses import dataclass, field
from typing import Optional
import logging
import numpy as np

from videounit.perception.sam2 import SAM2Segmenter
from videounit.perception.cotracker3 import CoTrackerTracker
from videounit.perception.midas import DepthEstimator

log = logging.getLogger(__name__)


@dataclass
class PerceptionResult:
    """Result from perception pipeline."""
    frames_dir: str
    segmentation: list[dict]  # Per-frame segmentation masks
    tracking: dict  # CoTracker3 tracks
    depth: list[np.ndarray]  # Depth maps
    metadata: dict = field(default_factory=dict)


class PerceptionPipeline:
    """Wrapper for AETHER perception pipeline.

    Coordinates SAM2 segmentation, CoTracker3 tracking, and MiDaS depth estimation.
    Can also delegate to AETHER's AetherNeuralCore when available.
    """

    def __init__(
        self,
        device: str = "cuda",
        use_aether_backend: bool = True,
        sam2_points_per_side: int = 16,
        cotracker_window_len: int = 16,
    ):
        """Initialize perception pipeline.

        Args:
            device: Compute device ('cuda' or 'cpu').
            use_aether_backend: Whether to prefer AETHER backend when available.
            sam2_points_per_side: Grid density for SAM2 segmentation.
            cotracker_window_len: Window length for CoTracker3.
        """
        self.device = device
        self.use_aether_backend = use_aether_backend
        self._sam2 = SAM2Segmenter(points_per_side=sam2_points_per_side)
        self._cotracker = CoTrackerTracker(window_len=cotracker_window_len)
        self._depth = DepthEstimator()
        self._aether_core = None

    def _load_aether_core(self):
        """Lazy load AETHER neural core."""
        if self._aether_core is not None:
            return
        try:
            from backend.app.perception.optimized.pipeline import AetherNeuralCore
            self._aether_core = AetherNeuralCore()
            log.info("AETHER AetherNeuralCore loaded")
        except ImportError:
            log.debug("AETHER backend not available, using native implementations")
            self._aether_core = None

    async def run(self, frames_dir: str) -> PerceptionResult:
        """Run full perception pipeline on extracted frames.

        Args:
            frames_dir: Directory containing extracted video frames.

        Returns:
            PerceptionResult with segmentation, tracking, depth, and metadata.
        """
        if self.use_aether_backend:
            self._load_aether_core()
            if self._aether_core is not None:
                return await self._run_aether(frames_dir)

        return await self._run_native(frames_dir)

    async def _run_aether(self, frames_dir: str) -> PerceptionResult:
        """Run perception via AETHER backend."""
        result = await self._aether_core.run(frames_dir)
        return PerceptionResult(
            frames_dir=frames_dir,
            segmentation=result.get("segmentation", []),
            tracking=result.get("tracking", {}),
            depth=result.get("depth", []),
            metadata=result.get("metadata", {}),
        )

    async def _run_native(self, frames_dir: str) -> PerceptionResult:
        """Run perception using native SAM2, CoTracker3, MiDaS."""
        import asyncio
        from pathlib import Path

        frames_path = Path(frames_dir)
        frame_files = sorted(
            [f for f in frames_path.glob("*.jpg")] +
            [f for f in frames_path.glob("*.png")],
            key=lambda f: f.name
        )

        if not frame_files:
            frame_files = sorted(frames_path.glob("*"), key=lambda f: f.name)

        import cv2
        frames = []
        for fpath in frame_files:
            frame = cv2.imread(str(fpath))
            if frame is not None:
                frames.append(frame)

        if not frames:
            log.warning(f"No frames found in {frames_dir}")
            return PerceptionResult(
                frames_dir=frames_dir,
                segmentation=[],
                tracking={"tracks": [], "frame_count": 0, "track_count": 0},
                depth=[],
                metadata={"frame_count": 0},
            )

        log.info(f"Processing {len(frames)} frames with native perception")

        segmentations = []
        for i, frame in enumerate(frames):
            masks = self._sam2.generate(frame)
            for mask in masks:
                mask["frame_idx"] = i
            segmentations.append({"frame_idx": i, "masks": masks})

        tracking = self._cotracker.track(frames)

        depth_maps = self._depth.estimate_batch(frames)

        metadata = {
            "frame_count": len(frames),
            "sam2_points_per_side": self._sam2._points_per_side,
            "cotracker_window_len": self._cotracker._window_len,
            "device": self.device,
        }

        return PerceptionResult(
            frames_dir=frames_dir,
            segmentation=segmentations,
            tracking=tracking,
            depth=depth_maps,
            metadata=metadata,
        )

    def unload(self):
        """Release model resources."""
        self._sam2.unload()
        self._cotracker.unload()
        self._depth.unload()
        self._aether_core = None
