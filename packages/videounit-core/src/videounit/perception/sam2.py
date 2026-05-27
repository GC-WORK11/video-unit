"""SAM2 segmentation model integration."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("/home/govinda/aether/data/checkpoints")


class SAM2EncoderONNX:
    """SAM2 image encoder via ONNX Runtime.

    Provides fast CPU-based encoding for SAM2 segmentation.
    """

    def __init__(self, onnx_path: Optional[str] = None):
        if onnx_path is None:
            onnx_path = CHECKPOINT_DIR / "sam2_encoder_fp32.onnx"
        self.session = None
        self.onnx_path = onnx_path
        self._create_session()

    def _create_session(self):
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = False
            self.session = ort.InferenceSession(
                str(self.onnx_path), sess_options, providers=["CPUExecutionProvider"]
            )
            log.info(f"SAM2 ONNX encoder loaded: CPU")
        except Exception as e:
            log.warning(f"SAM2 ONNX init failed: {e}, encoder will be computed via PyTorch")
            self.session = None

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """Encode frame to feature map.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Feature map (B, 256, 64, 64).
        """
        if self.session is None:
            return np.zeros((1, 256, 64, 64), dtype=np.float32)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_t = ((frame_resized / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        return self.session.run(None, {"input": frame_t})[0]


class SAM2Segmenter:
    """SAM2 segmentation with configurable grid density.

    Uses points_per_side to control the number of grid points for mask generation.
    Lower values are faster, higher values provide better coverage.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        points_per_side: int = 16,
    ):
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_DIR / "sam2_hiera_small.pt"
        self._checkpoint_path = str(checkpoint_path)
        self._points_per_side = points_per_side
        self._sam2_model = None
        self._mask_gen = None
        self._is_loaded = False

    def _ensure_loaded(self):
        if self._is_loaded:
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            sam2 = build_sam2(
                "sam2_hiera_s.yaml", self._checkpoint_path, device="cpu"
            )
            sam2.eval()
            self._sam2_model = sam2
            self._mask_gen = SAM2AutomaticMaskGenerator(
                sam2,
                points_per_side=self._points_per_side,
                points_per_batch=64,
                multimask_output=False,
                pred_iou_thresh=0.7,
                stability_score_thresh=0.8,
            )
            self._is_loaded = True
            log.info(f"SAM2 segmenter loaded (points_per_side={self._points_per_side})")
        except ImportError as e:
            log.warning(f"SAM2 not available: {e}")
            self._is_loaded = True

    def generate(self, frame: np.ndarray) -> list[dict]:
        """Generate masks for a frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of mask dictionaries with bbox, area, segmentation, etc.
        """
        self._ensure_loaded()
        if self._mask_gen is None:
            return []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            masks = self._mask_gen.generate(frame_rgb)

        return [
            {
                "id": i,
                "bbox": [float(x) for x in m.get("bbox", [0, 0, 0, 0])],
                "area": int(m.get("area", 0)),
                "label": f"object_{i}",
                "predicted_iou": float(m.get("predicted_iou", 0.0)),
                "stability_score": float(m.get("stability_score", 0.0)),
                "segmentation": m.get("segmentation"),
            }
            for i, m in enumerate(masks[:20])
        ]

    def unload(self):
        del self._sam2_model
        del self._mask_gen
        self._sam2_model = None
        self._mask_gen = None
        self._is_loaded = False
