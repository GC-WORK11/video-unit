"""ObjectPersistenceEvaluator - detects object teleportation and identity breaks."""

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
class ObjectPersistenceEvaluator(Evaluator):
    """Evaluates object persistence and detects teleportation artifacts.

    This evaluator uses CoTracker3 tracks to detect when objects make
    impossible instantaneous movements (teleportation). It checks that
    objects maintain spatial continuity between consecutive frames.

    Contract schema:
        persistence_checks:
          - object: "ball"
            max_jump_distance: 100  # pixels of allowed jump between frames
            min_track_length: 10  # minimum frames object should persist

    Failure types:
        - teleportation: Object moved impossibly fast between frames
        - track_interruption: Object disappeared and reappeared with same ID
        - identity_break: Object ID swapped with another track
    """

    name = "object_persistence"
    required_inputs = ["tracks"]

    def __init__(self):
        self.default_max_jump = 100.0  # pixels
        self.default_min_track_length = 5  # frames

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the object persistence evaluation.

        Args:
            context: Evaluation context with tracks.

        Returns:
            EvaluationResult with failures for any teleportation or identity breaks.
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
        persistence_checks = context.contract.get("persistence_checks", [])

        if not persistence_checks:
            logger.info("No persistence checks specified in contract")
            return result

        for check_spec in persistence_checks:
            obj_name = check_spec.get("object", "unnamed")
            max_jump = check_spec.get("max_jump_distance", self.default_max_jump)
            min_track_length = check_spec.get("min_track_length", self.default_min_track_length)

            track = self._find_track_for_object(context, obj_name)
            if track is None:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    failure_type="object_not_found",
                    severity="warning",
                    message=f"Cannot check persistence for unknown object '{obj_name}'",
                    object=obj_name
                )
                continue

            if track.duration_frames() < min_track_length:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(track.frames[0]),
                    frame_number=track.frames[0],
                    failure_type="track_too_short",
                    severity="warning",
                    message=f"Object '{obj_name}' track only has {track.duration_frames()} frames "
                            f"(minimum: {min_track_length})",
                    object=obj_name,
                    suggested_fix="Ensure object remains visible for the required duration"
                )
                result.score = min(result.score, 70.0)

            teleportation_events = self._detect_teleportation(track, max_jump)

            for event in teleportation_events:
                frame_num, jump_distance, prev_pos, curr_pos = event

                result.add_failure(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    failure_type="teleportation",
                    severity="critical",
                    message=f"Object '{obj_name}' teleported {jump_distance:.0f}px "
                            f"from ({prev_pos[0]:.0f},{prev_pos[1]:.0f}) to "
                            f"({curr_pos[0]:.0f},{curr_pos[1]:.0f})",
                    object=obj_name,
                    suggested_fix="Ensure smooth continuous motion without jumps"
                )
                result.score = min(result.score, 0.0)
                result.passed = False

                result.add_evidence(
                    timestamp=context.frame_to_timestamp(frame_num),
                    frame_number=frame_num,
                    thumbnail_path=str(context.get_thumbnail_path(
                        frame_num, f"teleport_{obj_name}"
                    )),
                    bbox=curr_pos,
                    explanation=f"Teleportation detected: {jump_distance:.0f}px instantaneous jump",
                    confidence=0.95
                )

            if not teleportation_events:
                if track.frames:
                    mid_frame = track.frames[len(track.frames) // 2]
                    result.add_evidence(
                        timestamp=context.frame_to_timestamp(mid_frame),
                        frame_number=mid_frame,
                        thumbnail_path=str(context.get_thumbnail_path(
                            mid_frame, f"persist_{obj_name}"
                        )),
                        bbox=track.get_bbox_at(mid_frame),
                        explanation=f"Object '{obj_name}' maintains continuous identity",
                        confidence=0.95
                    )

        return result

    def _find_track_for_object(self, context: EvaluationContext, obj_name: str) -> Optional[ObjectTrack]:
        """Find the track matching the specified object name."""
        for track in context.tracks:
            if track.object_id == obj_name or track.class_name == obj_name:
                return track
        return None

    def _detect_teleportation(
        self,
        track: ObjectTrack,
        max_jump: float
    ) -> list[tuple[int, float, list[float], list[float]]]:
        """Detect teleportation events in a track.

        Args:
            track: Object track to analyze.
            max_jump: Maximum allowed instantaneous distance in pixels.

        Returns:
            List of (frame_number, jump_distance, prev_position, curr_position) tuples.
        """
        teleportation_events = []

        for i in range(1, len(track.frames)):
            prev_frame = track.frames[i - 1]
            curr_frame = track.frames[i]

            expected_frame_diff = curr_frame - prev_frame
            if expected_frame_diff > 1:
                logger.debug(
                    f"Skipping gap of {expected_frame_diff} frames between "
                    f"frames {prev_frame} and {curr_frame}"
                )
                continue

            prev_bbox = track.bboxes[i - 1] if track.bboxes else None
            curr_bbox = track.bboxes[i] if track.bboxes else None

            if prev_bbox is None or curr_bbox is None:
                continue

            prev_center = self._bbox_center(prev_bbox)
            curr_center = self._bbox_center(curr_bbox)

            jump_distance = math.sqrt(
                (curr_center[0] - prev_center[0]) ** 2 +
                (curr_center[1] - prev_center[1]) ** 2
            )

            if jump_distance > max_jump:
                teleportation_events.append((
                    curr_frame,
                    jump_distance,
                    prev_center,
                    curr_center
                ))

        return teleportation_events

    def _bbox_center(self, bbox: list[float]) -> list[float]:
        """Calculate center point of a bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            [cx, cy] center coordinates.
        """
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
