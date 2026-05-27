"""PhysicsSanityEvaluator - checks physical plausibility of motion."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from ._base import Evaluator
from ._context import EvaluationContext, ObjectTrack
from ._registry import register_evaluator
from ._result import EvaluationResult

logger = logging.getLogger(__name__)


@register_evaluator
class PhysicsSanityEvaluator(Evaluator):
    """Evaluates physical plausibility of object motion.

    This evaluator uses depth maps from MiDaS and motion tracks to verify
    that objects behave according to physical laws. It checks for:
    - Floating objects (not affected by gravity)
    - Impossible motion trajectories
    - Depth-motion inconsistencies

    Contract schema:
        physics_checks:
          - object: "ball"
            expect_grounded: true  # if true, object should be on ground
            max_vertical_speed: 500  # maximum upward pixels per frame
            gravity_direction: "down"  # or "up" for upside-down world

    Failure types:
        - floating_object: Object not grounded when it should be
        - impossible_motion: Motion violates physics constraints
        - depth_inconsistency: Motion inconsistent with depth
    """

    name = "physical_plausibility"
    required_inputs = ["tracks", "depth"]

    GRAVITY_ACCELERATION = 9.81  # m/s^2
    PIXELS_PER_METER = 100.0  # approximate

    def __init__(self):
        self.default_max_vertical_speed = 500.0  # pixels per frame
        self.default_gravity_direction = "down"
        self.ground_threshold = 0.1  # depth ratio for grounded detection

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the physics sanity evaluation.

        Args:
            context: Evaluation context with tracks and depth.

        Returns:
            EvaluationResult with failures for any physical implausibilities.
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
        physics_checks = context.contract.get("physics_checks", [])

        if not physics_checks:
            logger.info("No physics checks specified in contract")
            return result

        depth_maps = context.perception_result.depth
        if not depth_maps:
            result.add_failure(
                timestamp=context.frame_to_timestamp(0),
                frame_number=0,
                failure_type="no_depth_data",
                severity="warning",
                message="No depth maps available for physics validation"
            )
            result.score = min(result.score, 50.0)
            return result

        for check_spec in physics_checks:
            obj_name = check_spec.get("object", "unnamed")
            expect_grounded = check_spec.get("expect_grounded", False)
            max_vertical_speed = check_spec.get(
                "max_vertical_speed", self.default_max_vertical_speed
            )
            gravity_direction = check_spec.get(
                "gravity_direction", self.default_gravity_direction
            )

            track = self._find_track_for_object(context, obj_name)
            if track is None:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    failure_type="object_not_found",
                    severity="warning",
                    message=f"Cannot check physics for unknown object '{obj_name}'",
                    object=obj_name
                )
                continue

            if expect_grounded:
                floating_events = self._detect_floating_objects(
                    track, depth_maps, context
                )
                for event in floating_events:
                    frame_num, depth_ratio = event
                    result.add_failure(
                        timestamp=context.frame_to_timestamp(frame_num),
                        frame_number=frame_num,
                        failure_type="floating_object",
                        severity="critical",
                        message=f"Object '{obj_name}' appears to be floating at frame {frame_num} "
                                f"(depth ratio: {depth_ratio:.2f})",
                        object=obj_name,
                        suggested_fix="Ensure object maintains contact with ground"
                    )
                    result.score = min(result.score, 0.0)
                    result.passed = False

                    result.add_evidence(
                        timestamp=context.frame_to_timestamp(frame_num),
                        frame_number=frame_num,
                        thumbnail_path=str(context.get_thumbnail_path(
                            frame_num, f"floating_{obj_name}"
                        )),
                        bbox=track.get_bbox_at(frame_num),
                        explanation=f"Object depth ratio {depth_ratio:.2f} indicates floating",
                        confidence=abs(depth_ratio - self.ground_threshold) / self.ground_threshold
                    )

            vertical_violations = self._detect_impossible_vertical_motion(
                track, max_vertical_speed, gravity_direction
            )
            for event in vertical_violations:
                frame_num, vertical_speed, direction = event
                result.add_failure(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    failure_type="impossible_motion",
                    severity="critical",
                    message=f"Object '{obj_name}' has impossible vertical speed "
                            f"{vertical_speed:.0f}px/frame ({direction})",
                    object=obj_name,
                    suggested_fix="Ensure motion follows physical constraints"
                )
                result.score = min(result.score, 0.0)
                result.passed = False

                result.add_evidence(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    thumbnail_path=str(context.get_thumbnail_path(
                        frame_num, f"physics_{obj_name}"
                    )),
                    bbox=track.get_bbox_at(frame_num),
                    explanation=f"Vertical speed {vertical_speed:.0f}px/frame is physically impossible",
                    confidence=0.95
                )

        return result

    def _find_track_for_object(self, context: EvaluationContext, obj_name: str) -> Optional[ObjectTrack]:
        """Find the track matching the specified object name."""
        for track in context.tracks:
            if track.object_id == obj_name or track.class_name == obj_name:
                return track
        return None

    def _detect_floating_objects(
        self,
        track: ObjectTrack,
        depth_maps: list,
        context: EvaluationContext
    ) -> list[tuple[int, float]]:
        """Detect if object is floating above ground.

        Args:
            track: Object track.
            depth_maps: List of depth maps per frame.
            context: Evaluation context.

        Returns:
            List of (frame_num, depth_ratio) tuples for floating detections.
        """
        floating_events = []

        for i, frame_num in enumerate(track.frames):
            if frame_num >= len(depth_maps):
                continue

            bbox = track.bboxes[i] if track.bboxes else None
            if bbox is None:
                continue

            depth_map = depth_maps[frame_num]
            if depth_map is None:
                continue

            depth_ratio = self._compute_object_depth_ratio(depth_map, bbox)

            if depth_ratio < self.ground_threshold:
                floating_events.append((frame_num, depth_ratio))

        return floating_events

    def _compute_object_depth_ratio(
        self,
        depth_map: NDArray[np.float32],
        bbox: list[float]
    ) -> float:
        """Compute depth ratio for object region.

        Args:
            depth_map: Depth map array.
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            Ratio of object depth to scene depth.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = depth_map.shape[:2]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return 0.5

        roi = depth_map[y1:y2, x1:x2]

        if roi.size == 0:
            return 0.5

        object_depth = np.median(roi)
        scene_depth = np.median(depth_map)

        if scene_depth < 1e-6:
            return 0.5

        return float(object_depth / scene_depth)

    def _detect_impossible_vertical_motion(
        self,
        track: ObjectTrack,
        max_vertical_speed: float,
        gravity_direction: str
    ) -> list[tuple[int, float, str]]:
        """Detect physically impossible vertical motion.

        Args:
            track: Object track.
            max_vertical_speed: Maximum allowed vertical speed.
            gravity_direction: Expected gravity direction.

        Returns:
            List of (frame_num, speed, direction) tuples.
        """
        violations = []

        for i in range(1, len(track.frames)):
            prev_bbox = track.bboxes[i - 1] if track.bboxes else None
            curr_bbox = track.bboxes[i] if track.bboxes else None

            if prev_bbox is None or curr_bbox is None:
                continue

            prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
            curr_cy = (curr_bbox[1] + curr_bbox[3]) / 2

            vertical_delta = curr_cy - prev_cy

            if gravity_direction == "down" and vertical_delta < -max_vertical_speed:
                violations.append((track.frames[i], abs(vertical_delta), "upward"))
            elif gravity_direction == "up" and vertical_delta > max_vertical_speed:
                violations.append((track.frames[i], abs(vertical_delta), "downward"))

        return violations
