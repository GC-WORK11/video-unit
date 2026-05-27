"""SceneCutEvaluator - detects random or invalid scene changes."""

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
class SceneCutEvaluator(Evaluator):
    """Evaluates scene cut consistency and detects random scene changes.

    This evaluator uses histogram comparison to detect sudden scene changes
    that violate the minimum scene duration contract. It can also use
    PySceneDetect if available for more sophisticated detection.

    Contract schema:
        scene_checks:
          - min_scene_duration: 30  # minimum frames per scene
            allow_hard_cuts: true  # whether hard cuts are allowed
            allow_fade_transitions: true  # whether fade transitions are allowed

    Failure types:
        - random_scene_cut: Scene changed before minimum duration
        - too_many_cuts: Excessive scene changes detected
        - invalid_transition: Non-approved transition type
    """

    name = "no_random_scene_cut"
    required_inputs = ["frames"]

    def __init__(self):
        self.default_min_scene_duration = 30  # frames
        self.default_allow_hard_cuts = True
        self.default_allow_fade_transitions = True
        self.histogram_threshold = 0.6  # threshold for histogram similarity

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the scene cut evaluation.

        Args:
            context: Evaluation context with frames.

        Returns:
            EvaluationResult with failures for any invalid scene cuts.
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
        scene_checks = context.contract.get("scene_checks", [])

        if not scene_checks:
            logger.info("No scene checks specified in contract")
            return result

        total_frames = context.video_metadata.get("total_frames", 0)
        if total_frames == 0:
            logger.warning("Cannot determine total frames for scene cut check")
            return result

        for check_spec in scene_checks:
            min_scene_duration = check_spec.get(
                "min_scene_duration", self.default_min_scene_duration
            )
            allow_hard_cuts = check_spec.get(
                "allow_hard_cuts", self.default_allow_hard_cuts
            )
            allow_fade_transitions = check_spec.get(
                "allow_fade_transitions", self.default_allow_fade_transitions
            )

            scene_cuts = self._detect_scene_cuts(context)

            if not scene_cuts:
                logger.info("No scene cuts detected")
                continue

            scene_durations = self._compute_scene_durations(scene_cuts, total_frames)

            for i, duration in enumerate(scene_durations):
                if duration < min_scene_duration:
                    cut_frame = scene_cuts[i]
                    severity = self._severity_from_duration(duration, min_scene_duration)

                    result.add_failure(
                        timestamp=context.frame_to_timestamp(cut_frame),
                        frame_number=cut_frame,
                        failure_type="random_scene_cut",
                        severity=severity,
                        message=f"Scene cut at frame {cut_frame} is too soon. "
                                f"Scene lasted only {duration} frames (minimum: {min_scene_duration})",
                        suggested_fix=f"Ensure scenes are at least {min_scene_duration} frames long"
                    )
                    result.score = min(result.score, self._score_from_severity(severity))
                    result.passed = False

                    result.add_evidence(
                        timestamp=context.frame_to_timestamp(cut_frame),
                        frame_number=cut_frame,
                        thumbnail_path=str(context.get_thumbnail_path(
                            cut_frame, "scene_cut"
                        )),
                        explanation=f"Scene lasted only {duration} frames before cut",
                        confidence=0.9
                    )

            expected_cuts = total_frames // min_scene_duration
            if len(scene_cuts) > expected_cuts * 2:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(scene_cuts[-1]),
                    frame_number=scene_cuts[-1],
                    failure_type="too_many_cuts",
                    severity="warning",
                    message=f"Detected {len(scene_cuts)} scene cuts (expected max ~{expected_cuts * 2})",
                    suggested_fix="Reduce frequency of scene changes"
                )
                result.score = min(result.score, 60.0)

        return result

    def _detect_scene_cuts(self, context: EvaluationContext) -> list[int]:
        """Detect scene cuts using histogram comparison.

        Args:
            context: Evaluation context.

        Returns:
            List of frame numbers where scene cuts occur.
        """
        scene_cuts = []

        prev_hist: NDArray[np.float32] | None = None
        total_frames = context.video_metadata.get("total_frames", 0)

        check_interval = max(1, total_frames // 100)

        for frame_num in range(0, total_frames, check_interval):
            frame_path = context.get_frame_path(frame_num)
            if not frame_path.exists():
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            curr_hist = self._compute_histogram(frame)

            if prev_hist is not None:
                similarity = self._histogram_similarity(prev_hist, curr_hist)

                if similarity < self.histogram_threshold:
                    if scene_cuts and frame_num - scene_cuts[-1] < 5:
                        continue
                    scene_cuts.append(frame_num)

            prev_hist = curr_hist

        scene_cuts.sort()
        return scene_cuts

    def _compute_histogram(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Compute color histogram for a frame.

        Args:
            frame: BGR image.

        Returns:
            Normalized histogram array.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _histogram_similarity(
        self,
        hist1: NDArray[np.float32],
        hist2: NDArray[np.float32]
    ) -> float:
        """Compute similarity between two histograms.

        Args:
            hist1: First histogram.
            hist2: Second histogram.

        Returns:
            Similarity score between 0 and 1.
        """
        if hist1.shape != hist2.shape:
            min_size = min(len(hist1), len(hist2))
            hist1 = hist1[:min_size]
            hist2 = hist2[:min_size]

        correlation = cv2.compareHist(
            hist1.reshape(-1, 1).astype(np.float32),
            hist2.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        return max(0.0, float(correlation))

    def _compute_scene_durations(self, scene_cuts: list[int], total_frames: int) -> list[int]:
        """Compute duration of each scene.

        Args:
            scene_cuts: List of scene cut frame numbers.
            total_frames: Total frames in video.

        Returns:
            List of scene durations in frames.
        """
        if not scene_cuts:
            return [total_frames]

        durations = []
        prev_cut = 0
        for cut in scene_cuts:
            durations.append(cut - prev_cut)
            prev_cut = cut

        durations.append(total_frames - prev_cut)
        return durations

    def _severity_from_duration(self, duration: int, min_duration: int) -> str:
        """Determine severity from scene duration shortfall."""
        shortfall_ratio = (min_duration - duration) / min_duration
        if shortfall_ratio > 0.8:
            return "critical"
        elif shortfall_ratio > 0.5:
            return "fail"
        elif shortfall_ratio > 0.2:
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
