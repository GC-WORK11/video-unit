"""MiDaS depth estimation integration."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("/home/govinda/aether/data/checkpoints")


class DepthEstimator:
    """MiDaS depth estimation for monocular depth cues.

    Provides depth maps from single images using MiDaS DPT model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "DPT_Large",
    ):
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "dpt_large.pt"
        self._checkpoint_path = str(checkpoint_path)
        self._model_type = model_type
        self._model = None
        self._transform = None
        self._is_loaded = False

    def _ensure_loaded(self):
        if self._is_loaded:
            return
        try:
            import timm

            self._model = timm.create_model(
                f"midas_{self._model_type.lower()}",
                pretrained=True,
                checkpoint_path=self._checkpoint_path,
            )
            self._model.eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            self._transform = timm.data.create_transform(
                input_size=384,
                is_training=False,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            )
            self._is_loaded = True
            log.info(f"MiDaS {self._model_type} loaded")
        except ImportError as e:
            log.warning(f"MiDaS not available: {e}")
            self._is_loaded = True

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth map for a frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Depth map (H, W) with values normalized to [0, 1].
        """
        self._ensure_loaded()
        if self._model is None:
            h, w = frame.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self._transform(frame_rgb).unsqueeze(0)

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()

        with torch.no_grad():
            depth = self._model(input_tensor)

        depth = depth.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth.astype(np.float32)

    def estimate_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Estimate depth maps for multiple frames.

        Args:
            frames: List of BGR images.

        Returns:
            List of depth maps.
        """
        return [self.estimate(f) for f in frames]

    def unload(self):
        if self._model is not None:
            del self._model
        self._model = None
        self._transform = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
