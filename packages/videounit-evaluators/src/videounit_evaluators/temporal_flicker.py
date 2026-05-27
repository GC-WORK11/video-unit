"""TemporalFlickerEvaluator - detects frame-to-frame flickering artifacts."""

import logging
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ._base import Evaluator
from ._context import EvaluationContext
from ._registry import register_evaluator
from ._result import EvaluationResult

logger = logging.getLogger(__name__)


@register_evaluator
class TemporalFlickerEvaluator(Evaluator):
    """Evaluates temporal consistency and detects flickering artifacts.

    This evaluator computes frame-to-frame differences using SSIM (Structural
    Similarity Index) and pixel-based brightness comparison to detect sudden
    changes that snap back, indicating temporal instability.

    Contract schema:
        flicker_checks:
          - region: "full"  # or "object:ball" for specific object
            max_brightness_delta: 0.15  # 0-1 range for acceptable flicker
            min_ssim: 0.85  # minimum SSIM between consecutive frames
            check_interval: 5  # check every N frames

    Failure types:
        - brightness_flicker: Sudden brightness change exceeding threshold
        - ssim_drop: Structural similarity drops below threshold
        - position_snap: Object position snaps back after brief displacement
    """

    name = "no_temporal_flicker"
    required_inputs = ["frames"]

    def __init__(self):
        self.default_max_brightness_delta = 0.15
        self.default_min_ssim = 0.85
        self.default_check_interval = 5

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the temporal flicker evaluation.

        Args:
            context: Evaluation context with frames.

        Returns:
            EvaluationResult with failures for any detected flickering.
        """
        missing = self.get_missing_inputs(context)
        if missing:
            result = EvaluationResult(passed=False, score=0.0)
            result.add_failure(
                timestamp=context.frame_to_timestamp(0),
                frame_number=0,
                failure_type="perception_unavailable",
                severity="warning",
                message=f"Evaluator '{self.name}' could not run: missing inputs {missing}. "
                        "Perception pipeline may not have produced required outputs.",
                object=None,
                suggested_fix="Check that perception pipeline ran successfully and produced tracks/masks/depth.",
            )
            return result
        self.validate_inputs(context)

        result = EvaluationResult(passed=True, score=100.0)
        flicker_checks = context.contract.get("flicker_checks", [])

        if not flicker_checks:
            logger.info("No flicker checks specified in contract")
            return result

        total_frames = context.video_metadata.get("total_frames", 0)
        if total_frames == 0:
            logger.warning("Cannot determine total frames for flicker check")
            return result

        for check_spec in flicker_checks:
            region = check_spec.get("region", "full")
            max_brightness_delta = check_spec.get(
                "max_brightness_delta", self.default_max_brightness_delta
            )
            min_ssim = check_spec.get("min_ssim", self.default_min_ssim)
            check_interval = check_spec.get(
                "check_interval", self.default_check_interval
            )

            flicker_events = self._detect_flicker(
                context,
                region,
                max_brightness_delta,
                min_ssim,
                check_interval
            )

            for event in flicker_events:
                frame_num, flicker_type, value, severity = event

                result.add_failure(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    failure_type=flicker_type,
                    severity=severity,
                    message=f"Temporal flicker detected at frame {frame_num}: "
                            f"{flicker_type} = {value:.3f}",
                    suggested_fix="Apply temporal consistency post-processing to video"
                )
                result.score = min(result.score, self._score_from_severity(severity))
                result.passed = False

                result.add_evidence(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    thumbnail_path=str(context.get_thumbnail_path(
                        frame_num, f"flicker_{flicker_type}"
                    )),
                    explanation=f"{flicker_type} flicker detected: {value:.3f}",
                    confidence=min(abs(value - 0.5) * 2, 1.0)
                )

        return result

    def _detect_flicker(
        self,
        context: EvaluationContext,
        region: str,
        max_brightness_delta: float,
        min_ssim: float,
        check_interval: int
    ) -> list[tuple[int, str, float, str]]:
        """Detect flickering in video frames.

        Args:
            context: Evaluation context.
            region: Region to check ("full" or "object:name").
            max_brightness_delta: Maximum acceptable brightness change.
            min_ssim: Minimum acceptable SSIM.
            check_interval: Frame interval for checking.

        Returns:
            List of (frame_num, flicker_type, value, severity) tuples.
        """
        flicker_events = []

        prev_frame: NDArray[np.uint8] | None = None
        prev_gray: NDArray[np.uint8] | None = None

        frames_to_check = list(range(0, context.video_metadata.get("total_frames", 0), check_interval))
        if frames_to_check and frames_to_check[-1] != context.video_metadata.get("total_frames", 0) - 1:
            frames_to_check.append(context.video_metadata.get("total_frames", 0) - 1)

        for frame_num in frames_to_check:
            frame_path = context.get_frame_path(frame_num)
            if not frame_path.exists():
                continue

            curr_frame = cv2.imread(str(frame_path))
            if curr_frame is None:
                continue

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                brightness_delta = self._compute_brightness_delta(prev_gray, curr_gray)
                ssim = self._compute_ssim(prev_gray, curr_gray)

                if brightness_delta > max_brightness_delta:
                    severity = self._severity_from_brightness(brightness_delta, max_brightness_delta)
                    flicker_events.append((
                        frame_num,
                        "brightness_flicker",
                        brightness_delta,
                        severity
                    ))

                if ssim < min_ssim:
                    severity = self._severity_from_ssim(ssim, min_ssim)
                    flicker_events.append((
                        frame_num,
                        "ssim_drop",
                        ssim,
                        severity
                    ))

            prev_gray = curr_gray

        return flicker_events

    def _compute_brightness_delta(
        self,
        prev_gray: NDArray[np.uint8],
        curr_gray: NDArray[np.uint8]
    ) -> float:
        """Compute normalized brightness difference between frames.

        Args:
            prev_gray: Previous grayscale frame.
            curr_gray: Current grayscale frame.

        Returns:
            Brightness delta in 0-1 range.
        """
        mean_prev = np.mean(prev_gray) / 255.0
        mean_curr = np.mean(curr_gray) / 255.0
        return abs(mean_curr - mean_prev)

    def _compute_ssim(
        self,
        prev_gray: NDArray[np.uint8],
        curr_gray: NDArray[np.uint8]
    ) -> float:
        """Compute Structural Similarity Index between frames.

        Args:
            prev_gray: Previous grayscale frame.
            curr_gray: Current grayscale frame.

        Returns:
            SSIM value between 0 and 1.
        """
        if prev_gray.shape != curr_gray.shape:
            curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(prev_gray.astype(np.float32), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(curr_gray.astype(np.float32), (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(prev_gray.astype(np.float32) ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(curr_gray.astype(np.float32) ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(prev_gray.astype(np.float32) * curr_gray.astype(np.float32), (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return float(np.mean(ssim_map))

    def _severity_from_brightness(self, delta: float, threshold: float) -> str:
        """Determine severity from brightness delta."""
        ratio = delta / threshold
        if ratio > 3.0:
            return "critical"
        elif ratio > 2.0:
            return "fail"
        elif ratio > 1.5:
            return "warning"
        else:
            return "info"

    def _severity_from_ssim(self, ssim: float, threshold: float) -> str:
        """Determine severity from SSIM value."""
        gap = threshold - ssim
        if gap > 0.1:
            return "critical"
        elif gap > 0.05:
            return "fail"
        elif gap > 0.02:
            return "warning"
        else:
            return "info"

    def _score_from_severity(self, severity: str) -> float:
        """Map severity to score penalty."""
        scores = {
            "critical": 0.0,
            "fail": 25.0,
            "warning": 50.0,
            "info": 80.0
        }
        return scores.get(severity, 50.0)
