"""ColorConsistencyEvaluator - checks if object colors remain consistent throughout video."""

import json
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
class ColorConsistencyEvaluator(Evaluator):
    """Evaluates color consistency of objects across video frames.

    This evaluator extracts the average color within object masks for each
    frame and detects significant color shifts. It uses LAB color space
    for perceptually accurate color comparison.

    Contract schema:
        color_checks:
          - object: "ball"
            tolerance: 30  # maximum acceptable color distance (0-100)
            check_gray: false  # if true, only checks brightness consistency

    Severity thresholds (color distance):
        - critical: distance > 2 * tolerance
        - fail: distance > tolerance
        - warning: distance > 0.5 * tolerance
    """

    name = "object_color_constant"
    required_inputs = ["masks", "frames"]

    # LAB color space difference thresholds for just-noticeable-difference
    JND_THRESHOLD = 30.0

    def __init__(self):
        self.tolerance = 30.0
        self.gray_mode = False

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the color consistency evaluation.

        Args:
            context: Evaluation context with masks and frames.

        Returns:
            EvaluationResult with failures for any significant color shifts.
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
        color_checks = context.contract.get("color_checks", [])

        if not color_checks:
            logger.info("No color checks specified in contract")
            return result

        total_frames = context.video_metadata.get("total_frames", 0)
        if total_frames == 0:
            logger.warning("Cannot determine total frames for color check")
            return result

        for check_spec in color_checks:
            obj_name = check_spec.get("object", "unnamed")
            tolerance = check_spec.get("tolerance", self.tolerance)
            check_gray = check_spec.get("check_gray", self.gray_mode)

            # Get the track for this object
            track = self._find_track_for_object(context, obj_name)
            if track is None:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    failure_type="object_not_found",
                    severity="warning",
                    message=f"Cannot check color for unknown object '{obj_name}'",
                    object=obj_name
                )
                continue

            # Analyze color consistency across track frames
            color_distances, frames_analyzed = self._analyze_color_consistency(
                context, track, check_gray
            )

            if not frames_analyzed:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(track.frames[0]),
                    frame_number=track.frames[0],
                    failure_type="no_color_data",
                    severity="warning",
                    message=f"No color data could be extracted for '{obj_name}'",
                    object=obj_name
                )
                continue

            # Find frames with significant color shifts
            shift_frames = self._detect_color_shifts(
                color_distances, frames_analyzed, tolerance
            )

            for frame_num, distance in shift_frames:
                severity = self._severity_from_distance(distance, tolerance)
                result.add_failure(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    failure_type="color_shift",
                    severity=severity,
                    message=f"Object '{obj_name}' has significant color shift "
                            f"(distance: {distance:.1f}, threshold: {tolerance})",
                    object=obj_name,
                    suggested_fix="Ensure consistent lighting throughout the video"
                )
                result.score = min(result.score, self._score_from_severity(severity))
                result.passed = False

                # Add evidence
                result.add_evidence(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    thumbnail_path=str(context.get_thumbnail_path(
                        frame_num, f"color_shift_{obj_name}"
                    )),
                    bbox=track.get_bbox_at(frame_num),
                    explanation=f"Color shift detected (distance: {distance:.1f})",
                    confidence=min(distance / (2 * tolerance), 1.0)
                )

        return result

    def _find_track_for_object(self, context: EvaluationContext, obj_name: str):
        """Find the track matching the specified object name."""
        for track in context.tracks:
            if track.object_id == obj_name or track.class_name == obj_name:
                return track
        return None

    def _analyze_color_consistency(
        self,
        context: EvaluationContext,
        track: object,
        check_gray: bool
    ) -> tuple[list[float], list[int]]:
        """Extract and compare colors across frames.

        Args:
            context: Evaluation context.
            track: Object track with bounding boxes.
            check_gray: If True, only check brightness (L channel).

        Returns:
            Tuple of (color_distances, frame_numbers) for each consecutive pair.
        """
        if not track.frames:
            return [], []

        colors: list[NDArray[np.float32]] = []
        frames_used: list[int] = []
        prev_color: NDArray[np.float32] | None = None
        distances: list[float] = []
        frames_with_distances: list[int] = []

        for i, frame_num in enumerate(track.frames):
            if i % max(1, len(track.frames) // 20) != 0 and i != len(track.frames) - 1:
                # Sample every Nth frame to reduce computation
                continue

            frame_path = context.get_frame_path(frame_num)
            if not frame_path.exists():
                continue

            bbox = track.bboxes[i] if track.bboxes else None
            color = self._extract_masked_color(frame_path, bbox)

            if color is None:
                continue

            colors.append(color)
            frames_used.append(frame_num)

            if prev_color is not None:
                if check_gray:
                    distance = float(abs(color[0] - prev_color[0]))
                else:
                    distance = self._lab_distance(color, prev_color)
                distances.append(distance)
                frames_with_distances.append(frame_num)

                prev_color = color
            else:
                prev_color = color

        return distances, frames_with_distances

    def _extract_masked_color(
        self, frame_path: Path, bbox: list[float] | None
    ) -> NDArray[np.float32] | None:
        """Extract average color from a masked region in a frame.

        Args:
            frame_path: Path to the frame image.
            bbox: Optional bounding box [x1, y1, x2, y2].

        Returns:
            LAB color values or None if extraction fails.
        """
        try:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                return None

            h, w = frame.shape[:2]

            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    return None
                roi = frame[y1:y2, x1:x2]
            else:
                # Use center region if no bbox
                margin_x, margin_y = w // 4, h // 4
                roi = frame[margin_y:h-margin_y, margin_x:w-margin_x]

            # Convert to LAB color space
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

            # Calculate mean color in LAB space
            mean_lab = cv2.mean(lab)

            return np.array(mean_lab[:3], dtype=np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract color from {frame_path}: {e}")
            return None

    def _lab_distance(self, lab1: NDArray[np.float32], lab2: NDArray[np.float32]) -> float:
        """Calculate perceptual color distance in LAB space.

        Uses simple Euclidean distance which approximates CIEDE2000
        for small distances.

        Args:
            lab1: First LAB color vector.
            lab2: Second LAB color vector.

        Returns:
            Color distance value.
        """
        diff = lab1 - lab2
        return float(np.sqrt(np.sum(diff ** 2)))

    def _detect_color_shifts(
        self,
        distances: list[float],
        frames: list[int],
        tolerance: float
    ) -> list[tuple[int, float]]:
        """Detect frames with color shifts exceeding tolerance.

        Args:
            distances: Color distances between consecutive frames.
            frames: Frame numbers corresponding to distances.
            tolerance: Maximum acceptable color distance.

        Returns:
            List of (frame_number, distance) tuples for exceeding frames.
        """
        shifts = []
        for frame_num, distance in zip(frames, distances):
            if distance > tolerance:
                shifts.append((frame_num, distance))
        return shifts

    def _severity_from_distance(self, distance: float, tolerance: float) -> str:
        """Determine severity based on how much tolerance was exceeded."""
        ratio = distance / tolerance
        if ratio > 2.0:
            return "critical"
        elif ratio > 1.5:
            return "fail"
        elif ratio > 1.0:
            return "warning"
        else:
            return "info"

    def _score_from_severity(self, severity: str) -> float:
        """Map severity to a score penalty."""
        scores = {
            "critical": 0.0,
            "fail": 30.0,
            "warning": 60.0,
            "info": 85.0
        }
        return scores.get(severity, 50.0)
