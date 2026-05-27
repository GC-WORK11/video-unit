"""Real perception pipeline - SAM 2 + CoTracker3 + MiDaS depth.
Memory-optimized: loads one model at a time, clears cache between stages."""
import gc
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from app.core.config import DATA_DIR

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("/home/govinda/aether/data/checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
LOW_VRAM = VRAM_GB < 6 if torch.cuda.is_available() else False

if LOW_VRAM:
    log.warning(f"Low VRAM detected ({VRAM_GB:.1f}GB) — will use CPU for heavy models")
    DEVICE = "cpu"

log.info(f"Perception pipeline using device: {DEVICE} (VRAM: {VRAM_GB:.1f}GB)")


def _clear_cuda_cache():
    """Free CUDA memory between model runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


class RealPerceptionPipeline:
    """Real perception pipeline — memory-efficient sequential loading."""

    def __init__(self):
        self._sam2 = None
        self._sam2_mask_gen = None
        self._cotracker = None
        self._midas = None

    def _get_sam2(self):
        """Lazy load SAM2."""
        if self._sam2 is None:
            sam2_ckpt = CHECKPOINT_DIR / "sam2_hiera_small.pt"
            if not sam2_ckpt.exists():
                raise FileNotFoundError(f"SAM2 checkpoint: {sam2_ckpt}")
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            log.info("Loading SAM2 model...")
            self._sam2 = build_sam2(
                "sam2_hiera_s.yaml",
                str(sam2_ckpt),
                device=DEVICE,
            )
            self._sam2_mask_gen = SAM2AutomaticMaskGenerator(self._sam2)
            log.info("SAM2 loaded")
        return self._sam2_mask_gen

    def _get_cotracker(self):
        """Lazy load CoTracker3."""
        if self._cotracker is None:
            ckpt = CHECKPOINT_DIR / "scaled_online.pth"
            if not ckpt.exists():
                raise FileNotFoundError(f"CoTracker checkpoint: {ckpt}")
            from cotracker.predictor import CoTrackerPredictor
            log.info("Loading CoTracker3...")
            self._cotracker = CoTrackerPredictor(
                checkpoint=str(ckpt),
                offline=True, v2=False, window_len=16,
            )
            self._cotracker = self._cotracker.to(DEVICE)
            self._cotracker.eval()
            log.info("CoTracker3 loaded")
        return self._cotracker

    def _get_midas(self):
        """Lazy load MiDaS."""
        if self._midas is None:
            log.info("Loading MiDaS...")
            self._midas = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS",
                pretrained=True, trust_repo=True,
            )
            self._midas = self._midas.to(DEVICE).eval()
            log.info("MiDaS loaded")
        return self._midas

    def segment_frame(self, frame: np.ndarray) -> dict:
        """Segment objects using SAM 2."""
        mask_gen = self._get_sam2()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masks = mask_gen.generate(frame_rgb)

        result_masks = []
        for i, m in enumerate(masks):
            seg = m.get("segmentation")
            if seg is None:
                continue
            bbox = m.get("bbox", [0, 0, 0, 0])
            result_masks.append({
                "id": i,
                "bbox": [float(x) for x in bbox],
                "area": int(m.get("area", 0)),
                "label": f"object_{i}",
                "predicted_iou": float(m.get("predicted_iou", 0.0)),
            })
            # Keep only first 20 masks to save memory
            if i >= 20:
                break

        return {"masks": result_masks, "count": len(result_masks), "method": "sam2", "device": DEVICE}

    def track_keypoints(self, frames: list[np.ndarray], grid_size: int = 6) -> dict:
        """Track keypoints using CoTracker3."""
        model = self._get_cotracker()

        # Subsample frames if too many
        max_frames = 15
        indices = np.linspace(0, len(frames) - 1, min(len(frames), max_frames), dtype=int)
        frames_sub = [frames[i] for i in indices]

        # CoTracker3: (B, T, 3, H, W), normalized [0,1]
        video = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_sub], axis=0)
        video = torch.from_numpy(video).float() / 255.0
        video = video.permute(0, 3, 1, 2)[None].to(DEVICE)  # (1, T, 3, H, W)

        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                video, grid_size=grid_size, backward_tracking=False,
            )

        tracks = []
        for f_idx in range(len(frames_sub)):
            frame_tracks = []
            for track_id in range(pred_tracks.shape[2]):
                frame_tracks.append({
                    "id": track_id,
                    "x": float(pred_tracks[0, f_idx, track_id, 0].cpu()),
                    "y": float(pred_tracks[0, f_idx, track_id, 1].cpu()),
                    "visibility": float(pred_visibility[0, f_idx, track_id].cpu()),
                })
            tracks.append(frame_tracks)

        return {
            "tracks": tracks,
            "frame_count": len(frames_sub),
            "track_count": int(pred_tracks.shape[2]),
            "method": "cotracker3",
            "device": DEVICE,
        }

    def estimate_depth(self, frame: np.ndarray) -> dict:
        """Estimate depth using MiDaS."""
        model = self._get_midas()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # MiDaS expects 384x384
        frame_small = cv2.resize(frame_rgb, (384, 384), interpolation=cv2.INTER_CUBIC)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        frame_norm = (frame_small / 255.0 - mean) / std
        frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).float()[None].to(DEVICE)

        with torch.no_grad():
            depth = model(frame_tensor)[0, 0].cpu().numpy()

        depth_full = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)

        return {
            "depth_map": depth_full,
            "min_depth": float(depth_full.min()),
            "max_depth": float(depth_full.max()),
            "mean_depth": float(depth_full.mean()),
            "method": "midas_v21",
            "device": DEVICE,
        }

    def run_full_pipeline(self, frames: list[np.ndarray]) -> dict:
        """Run complete perception pipeline — one model at a time to manage VRAM."""
        F = len(frames)
        log.info(f"Running perception on {F} frames...")

        # Step 1: SAM2 segmentation
        log.info("Step 1/3: SAM2 segmentation...")
        seg = self.segment_frame(frames[0])
        _clear_cuda_cache()

        # Step 2: CoTracker3 tracking
        log.info("Step 2/3: CoTracker3 tracking...")
        track = self.track_keypoints(frames)
        _clear_cuda_cache()

        # Step 3: MiDaS depth
        log.info("Step 3/3: MiDaS depth...")
        depth = self.estimate_depth(frames[0])

        return {
            "segmentation": seg,
            "tracking": track,
            "depth": depth,
            "frame_count": F,
            "device": DEVICE,
        }

    def unload_all(self):
        """Release all models from GPU memory."""
        for attr in ['_sam2', '_cotracker', '_midas']:
            model = getattr(self, attr, None)
            if model is not None:
                del model
                setattr(self, attr, None)
        _clear_cuda_cache()
        log.info("All models unloaded from GPU")


# Global singleton
_pipeline: RealPerceptionPipeline | None = None


def get_pipeline() -> RealPerceptionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RealPerceptionPipeline()
    return _pipeline
