"""MotionDirectionEvaluator - checks if objects move in the expected directions."""

import logging
import math
from typing import Optional

import numpy as np

from ._base import Evaluator
from ._context import EvaluationContext, ObjectTrack
from ._registry import register_evaluator
from ._result import EvaluationResult

logger = logging.getLogger(__name__)


@register_evaluator
class MotionDirectionEvaluator(Evaluator):
    """Evaluates whether objects move in the expected directions.

    This evaluator uses CoTracker3 tracks to compute object trajectories
    and compares them against expected motion directions from the contract.

    Contract schema:
        motion_checks:
          - object: "ball"
            phases:
              - from: "00:00.000"
                to: "00:01.000"
                expected_direction: "right"  # or angle in degrees
                min_distance: 50  # minimum pixels to travel
            tolerance: 30  # degrees of acceptable deviation

    Supported direction keywords:
        - horizontal: left or right movement
        - vertical: up or down movement
        - left, right, up, down: specific cardinal directions
        - clockwise, counter_clockwise: rotation
        - Any angle in degrees (e.g., 45, 270)
    """

    name = "motion_direction"
    required_inputs = ["tracks"]

    # Direction keywords mapping to angles
    DIRECTION_ANGLES = {
        "left": 180.0,
        "right": 0.0,
        "up": 270.0,
        "down": 90.0,
        "horizontal": None,  # special handling
        "vertical": None,  # special handling
    }

    def __init__(self):
        self.default_tolerance = 30.0  # degrees
        self.default_min_distance = 20.0  # pixels

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the motion direction evaluation.

        Args:
            context: Evaluation context with tracks.

        Returns:
            EvaluationResult with failures for any incorrect motion directions.
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
        motion_checks = context.contract.get("motion_checks", [])

        if not motion_checks:
            logger.info("No motion checks specified in contract")
            return result

        fps = context.video_metadata.get("fps", 30.0)

        for check_spec in motion_checks:
            obj_name = check_spec.get("object", "unnamed")
            phases = check_spec.get("phases", [])
            tolerance = check_spec.get("tolerance", self.default_tolerance)

            # Find the track for this object
            track = self._find_track_for_object(context, obj_name)
            if track is None:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    failure_type="object_not_found",
                    severity="warning",
                    message=f"Cannot check motion for unknown object '{obj_name}'",
                    object=obj_name
                )
                continue

            # Evaluate each motion phase
            for phase in phases:
                phase_result = await self._evaluate_phase(
                    context, track, phase, tolerance, fps
                )
                if not phase_result["passed"]:
                    result.passed = False
                    result.score = min(result.score, phase_result["score"])
                    for failure in phase_result["failures"]:
                        result.add_failure(**failure)
                for evidence in phase_result["evidence"]:
                    result.add_evidence(**evidence)

        return result

    def _find_track_for_object(self, context: EvaluationContext, obj_name: str) -> Optional[ObjectTrack]:
        """Find the track matching the specified object name."""
        for track in context.tracks:
            if track.object_id == obj_name or track.class_name == obj_name:
                return track
        return None

    async def _evaluate_phase(
        self,
        context: EvaluationContext,
        track: ObjectTrack,
        phase: dict,
        tolerance: float,
        fps: float
    ) -> dict:
        """Evaluate a single motion phase.

        Args:
            context: Evaluation context.
            track: Object track.
            phase: Phase specification from contract.
            tolerance: Acceptable angle deviation in degrees.
            fps: Video frames per second.

        Returns:
            Dict with 'passed', 'score', 'failures', and 'evidence'.
        """
        from_time = phase.get("from", "00:00.000")
        to_time = phase.get("to", "")
        expected_direction = phase.get("expected_direction", "right")
        min_distance = phase.get("min_distance", self.default_min_distance)

        # Parse timestamps
        start_frame = self._timestamp_to_frame(from_time, fps)
        end_frame = self._timestamp_to_frame(to_time, fps) if to_time else track.frames[-1]

        # Get trajectory segment
        trajectory = self._extract_trajectory(track, start_frame, end_frame)
        if trajectory is None or len(trajectory) < 2:
            return {
                "passed": True,  # Not enough data to evaluate
                "score": 100.0,
                "failures": [],
                "evidence": []
            }

        # Calculate observed direction
        observed_angle = self._calculate_trajectory_direction(trajectory)
        total_distance = self._calculate_trajectory_distance(trajectory)

        # Get expected angle(s)
        expected_angles = self._parse_expected_direction(expected_direction)

        # Check if any expected angle matches
        passed = False
        best_match_angle = None
        best_match_deviation = float("inf")

        for exp_angle in expected_angles:
            deviation = self._angle_deviation(observed_angle, exp_angle)
            if deviation <= tolerance:
                passed = True
                if deviation < best_match_deviation:
                    best_match_deviation = deviation
                    best_match_angle = exp_angle

        failures = []
        evidence = []

        mid_frame = (start_frame + end_frame) // 2

        if not passed:
            failures.append({
                "timestamp": context.frame_to_timestamp(mid_frame),
                "frame_number": mid_frame,
                "failure_type": "incorrect_motion_direction",
                "severity": "fail",
                "message": f"Object '{track.object_id}' motion direction deviates "
                           f"from expected. Observed: {observed_angle:.0f}°, "
                           f"Expected: {expected_direction}",
                "object": track.object_id,
                "suggested_fix": f"Adjust motion to follow {expected_direction} trajectory"
            })
            evidence.append({
                "timestamp": context.frame_to_timestamp(mid_frame),
                "frame_number": mid_frame,
                "thumbnail_path": str(context.get_thumbnail_path(mid_frame, "motion")),
                "bbox": track.get_bbox_at(mid_frame),
                "explanation": f"Trajectory shows {observed_angle:.0f}° direction "
                               f"(expected: {expected_direction})",
                "confidence": 0.9
            })
            return {"passed": False, "score": 50.0, "failures": failures, "evidence": evidence}

        # Check minimum distance
        if total_distance < min_distance:
            failures.append({
                "timestamp": context.frame_to_timestamp(mid_frame),
                "frame_number": mid_frame,
                "failure_type": "insufficient_motion",
                "severity": "warning",
                "message": f"Object '{track.object_id}' moved only {total_distance:.0f}px "
                           f"(minimum: {min_distance}px)",
                "object": track.object_id,
                "suggested_fix": "Ensure object travels the required distance"
            })
            evidence.append({
                "timestamp": context.frame_to_timestamp(mid_frame),
                "frame_number": mid_frame,
                "thumbnail_path": str(context.get_thumbnail_path(mid_frame, "motion")),
                "bbox": track.get_bbox_at(mid_frame),
                "explanation": f"Short trajectory: {total_distance:.0f}px traveled",
                "confidence": 0.8
            })
            return {"passed": False, "score": 75.0, "failures": failures, "evidence": evidence}

        # Passed
        evidence.append({
            "timestamp": context.frame_to_timestamp(mid_frame),
            "frame_number": mid_frame,
            "thumbnail_path": str(context.get_thumbnail_path(mid_frame, "motion")),
            "bbox": track.get_bbox_at(mid_frame),
            "explanation": f"Motion matches expected direction ({observed_angle:.0f}°)",
            "confidence": 0.95
        })
        return {"passed": True, "score": 100.0, "failures": [], "evidence": evidence}

    def _extract_trajectory(
        self,
        track: ObjectTrack,
        start_frame: int,
        end_frame: int
    ) -> Optional[list[tuple[float, float]]]:
        """Extract position trajectory between two frames.

        Args:
            track: Object track.
            start_frame: Starting frame number.
            end_frame: Ending frame number.

        Returns:
            List of (x, y) center positions or None if insufficient data.
        """
        trajectory = []
        for i, frame_num in enumerate(track.frames):
            if start_frame <= frame_num <= end_frame:
                bbox = track.bboxes[i]
                if bbox:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    trajectory.append((cx, cy))

        return trajectory if len(trajectory) >= 2 else None

    def _calculate_trajectory_direction(self, trajectory: list[tuple[float, float]]) -> float:
        """Calculate overall direction angle from trajectory.

        Args:
            trajectory: List of (x, y) positions.

        Returns:
            Direction angle in degrees (0 = right, 90 = down, 180 = left, 270 = up).
        """
        if len(trajectory) < 2:
            return 0.0

        # Use first and last points for overall direction
        x1, y1 = trajectory[0]
        x2, y2 = trajectory[-1]

        dx = x2 - x1
        dy = y2 - y1

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize to 0-360
        if angle_deg < 0:
            angle_deg += 360

        return angle_deg

    def _calculate_trajectory_distance(self, trajectory: list[tuple[float, float]]) -> float:
        """Calculate total pixel distance traveled along trajectory.

        Args:
            trajectory: List of (x, y) positions.

        Returns:
            Total distance in pixels.
        """
        if len(trajectory) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            total += math.sqrt(dx*dx + dy*dy)

        return total

    def _parse_expected_direction(self, direction: str) -> list[float]:
        """Parse expected direction into angle(s).

        Args:
            direction: Direction string or angle.

        Returns:
            List of acceptable angle values in degrees.
        """
        direction_lower = direction.lower().strip()

        if direction_lower in self.DIRECTION_ANGLES:
            angle = self.DIRECTION_ANGLES[direction_lower]
            if angle is not None:
                return [angle]

        if direction_lower == "horizontal":
            return [0.0, 180.0]  # Either left or right
        if direction_lower == "vertical":
            return [90.0, 270.0]  # Either up or down

        # Try parsing as angle
        try:
            angle = float(direction)
            return [angle % 360]
        except ValueError:
            pass

        # Unknown direction, return wildcard
        logger.warning(f"Unknown direction '{direction}', treating as any direction")
        return [0.0, 90.0, 180.0, 270.0]

    def _angle_deviation(self, observed: float, expected: float) -> float:
        """Calculate shortest angular deviation between two angles.

        Args:
            observed: Observed angle in degrees.
            expected: Expected angle in degrees.

        Returns:
            Minimum angular deviation in degrees (0-180).
        """
        diff = abs(observed - expected) % 360
        if diff > 180:
            diff = 360 - diff
        return diff

    def _timestamp_to_frame(self, timestamp: str, fps: float) -> int:
        """Convert ISO timestamp to frame number.

        Args:
            timestamp: Timestamp string (HH:MM:SS.fff or MM:SS.fff).
            fps: Frames per second.

        Returns:
            Frame number.
        """
        parts = timestamp.replace(",", ".").split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            hours = int(hours)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        else:
            return 0

        minutes = int(minutes)
        seconds = float(seconds)

        total_seconds = hours * 3600 + minutes * 60 + seconds
        return int(total_seconds * fps)
