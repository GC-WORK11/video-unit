"""CoTracker3 point tracking integration."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("/home/govinda/aether/data/checkpoints")


class CoTrackerTracker:
    """CoTracker3 point tracking across video frames.

    Tracks sparse points through video frames using optical flow.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, window_len: int = 16):
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "scaled_online.pth"
        self._checkpoint_path = str(checkpoint_path)
        self._window_len = window_len
        self._predictor = None
        self._is_loaded = False

    def _ensure_loaded(self):
        if self._is_loaded:
            return
        try:
            from cotracker.predictor import CoTrackerPredictor
            self._predictor = CoTrackerPredictor(
                checkpoint=self._checkpoint_path,
                offline=True,
                v2=False,
                window_len=self._window_len,
            )
            if torch.cuda.is_available():
                self._predictor = self._predictor.cuda()
            self._predictor.eval()
            self._is_loaded = True
            log.info("CoTracker3 loaded")
        except ImportError as e:
            log.warning(f"CoTracker3 not available: {e}")
            self._is_loaded = True

    def track(
        self,
        frames: list[np.ndarray],
        grid_size: int = 6,
    ) -> dict:
        """Track points across frames.

        Args:
            frames: List of BGR frames (H, W, 3).
            grid_size: Grid size for point initialization.

        Returns:
            Dictionary with tracks, frame_count, and track_count.
        """
        self._ensure_loaded()
        if self._predictor is None:
            return {"tracks": [], "frame_count": len(frames), "track_count": 0}

        T = min(len(frames), self._window_len)
        if len(frames) > T:
            indices = np.linspace(0, len(frames) - 1, T, dtype=int)
            frames_subset = [frames[i] for i in indices]
        else:
            frames_subset = frames

        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_subset]
        video = np.stack(frames_rgb, axis=0)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(0, 3, 1, 2)[None]

        if torch.cuda.is_available():
            video = video.cuda()

        with torch.no_grad():
            pred_tracks, pred_visibility = self._predictor(
                video, grid_size=grid_size, backward_tracking=False
            )

        tracks = []
        for f_idx in range(pred_tracks.shape[1]):
            frame_tracks = []
            for track_id in range(pred_tracks.shape[2]):
                x = float(pred_tracks[0, f_idx, track_id, 0].cpu())
                y = float(pred_tracks[0, f_idx, track_id, 1].cpu())
                vis = float(pred_visibility[0, f_idx, track_id].cpu())
                if vis > 0.1:
                    frame_tracks.append({
                        "id": track_id,
                        "x": x,
                        "y": y,
                        "visibility": vis,
                    })
            tracks.append(frame_tracks)

        return {
            "tracks": tracks,
            "frame_count": pred_tracks.shape[1],
            "track_count": int(pred_tracks.shape[2]),
        }

    def unload(self):
        del self._predictor
        self._predictor = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
