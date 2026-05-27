"""ObjectExistsEvaluator - checks if required objects exist throughout video duration."""

import logging
from typing import Optional

from ._base import Evaluator
from ._context import EvaluationContext
from ._registry import register_evaluator
from ._result import EvaluationResult

logger = logging.getLogger(__name__)


@register_evaluator
class ObjectExistsEvaluator(Evaluator):
    """Evaluates whether required objects exist throughout the video.

    This evaluator verifies that objects specified in the contract are
    present and visible for the required duration. It uses SAM2 masks
    from the perception pipeline to verify object presence.

    Contract schema:
        objects:
          - name: "ball"
            class: "sports_ball"
            required_duration: 1.0  # fraction of video (0.0-1.0)
            min_visibility: 0.8  # minimum fraction of frames object must be visible

    Severity thresholds:
        - critical: Object missing for >50% of required duration
        - fail: Object missing for 20-50% of required duration
        - warning: Object confidence below threshold but present
        - info: Object present with low confidence but acceptable
    """

    name = "object_exists"
    required_inputs = ["tracks", "masks"]

    def __init__(self):
        self.confidence_threshold = 0.5
        self.min_visibility_threshold = 0.8

    async def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run the object existence evaluation.

        Args:
            context: Evaluation context with tracks and masks.

        Returns:
            EvaluationResult with failures for any missing or low-confidence objects.
        """
        missing = self.get_missing_inputs(context)
        if missing:
            # Graceful degradation: missing perception data
            result = EvaluationResult(passed=False, score=0.0)
            contract_objects = context.contract.get("objects", [])
            for obj_spec in contract_objects:
                obj_name = obj_spec.get("name", "unnamed")
                result.add_failure(
                    timestamp=context.frame_to_timestamp(0),
                    frame_number=0,
                    failure_type="perception_unavailable",
                    severity="warning",
                    message=f"Object '{obj_name}' could not be evaluated: perception pipeline unavailable. "
                            f"Missing inputs: {missing}. Check GPU/CUDA availability.",
                    object=obj_name,
                    suggested_fix="Ensure CUDA is available and SAM2/CoTracker3 models can load. "
                                  "Perception pipeline requires GPU.",
                )
            return result

        result = EvaluationResult(passed=True, score=100.0)
        contract_objects = context.contract.get("objects", [])

        if not contract_objects:
            logger.info("No objects specified in contract, skipping object_exists check")
            return result

        total_frames = context.video_metadata.get("total_frames", 1)
        if total_frames == 0:
            total_frames = 1

        for obj_spec in contract_objects:
            obj_name = obj_spec.get("name", "unnamed")
            obj_class = obj_spec.get("class", "")
            required_duration = obj_spec.get("required_duration", 1.0)
            min_visibility = obj_spec.get("min_visibility", self.min_visibility_threshold)

            # Find matching track
            track = self._find_track_for_object(context, obj_name, obj_class)

            if track is None:
                self._add_missing_object_failure(
                    result, context, obj_name, obj_class,
                    "No track found for object"
                )
                continue

            # Analyze track coverage
            visible_frames = self._count_visible_frames(track, context)
            visibility_ratio = visible_frames / total_frames
            expected_min_frames = int(total_frames * required_duration * min_visibility)

            # Check average confidence
            avg_confidence = sum(track.confidences) / len(track.confidences) if track.confidences else 0.0

            if visible_frames < expected_min_frames:
                severity = self._determine_severity(
                    visibility_ratio, required_duration * min_visibility
                )
                result.add_failure(
                    timestamp=context.frame_to_timestamp(track.frames[0]),
                    frame_number=track.frames[0],
                    failure_type="object_missing",
                    severity=severity,
                    message=f"Object '{obj_name}' (class: {obj_class}) is only visible "
                            f"in {visible_frames}/{total_frames} frames ({visibility_ratio:.1%})",
                    object=obj_name,
                    suggested_fix=f"Ensure object remains in frame for at least "
                                  f"{required_duration * 100:.0f}% of the video"
                )
                result.score = min(result.score, 40.0 if severity == "critical" else 60.0)
                result.passed = False

            elif avg_confidence < self.confidence_threshold:
                result.add_failure(
                    timestamp=context.frame_to_timestamp(track.frames[0]),
                    frame_number=track.frames[0],
                    failure_type="low_confidence",
                    severity="warning",
                    message=f"Object '{obj_name}' has low detection confidence: {avg_confidence:.2f}",
                    object=obj_name,
                    suggested_fix="Improve lighting or contrast for better object detection"
                )
                result.score = min(result.score, 80.0)

            # Add evidence for the first and last appearances
            if track.frames:
                first_frame = track.frames[0]
                last_frame = track.frames[-1]

                result.add_evidence(
                    timestamp=context.frame_to_timestamp(first_frame),
                    frame_number=first_frame,
                    thumbnail_path=str(context.get_thumbnail_path(first_frame, f"obj_{obj_name}_start")),
                    bbox=track.bboxes[0] if track.bboxes else None,
                    explanation=f"Object '{obj_name}' first detected here",
                    confidence=track.confidences[0] if track.confidences else 1.0
                )

                result.add_evidence(
                    timestamp=context.frame_to_timestamp(last_frame),
                    frame_number=last_frame,
                    thumbnail_path=str(context.get_thumbnail_path(last_frame, f"obj_{obj_name}_end")),
                    bbox=track.bboxes[-1] if track.bboxes else None,
                    explanation=f"Object '{obj_name}' last detected here",
                    confidence=track.confidences[-1] if track.confidences else 1.0
                )

        return result

    def _find_track_for_object(
        self, context: EvaluationContext, obj_name: str, obj_class: str
    ) -> Optional[object]:
        """Find the track matching the specified object name or class.

        Args:
            context: Evaluation context.
            obj_name: Object name from contract.
            obj_class: Object class from contract.

        Returns:
            Matching ObjectTrack or None.
        """
        for track in context.tracks:
            # Match by name or class
            if obj_name and (track.object_id == obj_name or track.class_name == obj_name):
                return track
            if obj_class and track.class_name == obj_class:
                return track
        return None

    def _count_visible_frames(self, track: object, context: EvaluationContext) -> int:
        """Count how many frames the object is visible with sufficient confidence.

        Args:
            track: The object track.
            context: Evaluation context.

        Returns:
            Number of frames with confident detection.
        """
        count = 0
        for conf in track.confidences:
            if conf >= self.confidence_threshold:
                count += 1
        return count

    def _determine_severity(self, visibility_ratio: float, expected_ratio: float) -> str:
        """Determine failure severity based on visibility shortfall.

        Args:
            visibility_ratio: Actual visibility as fraction.
            expected_ratio: Expected visibility threshold.

        Returns:
            Severity string.
        """
        shortfall = expected_ratio - visibility_ratio
        if shortfall > 0.5:
            return "critical"
        elif shortfall > 0.2:
            return "fail"
        else:
            return "warning"

    def _add_missing_object_failure(
        self,
        result: EvaluationResult,
        context: EvaluationContext,
        obj_name: str,
        obj_class: str,
        reason: str
    ) -> None:
        """Add a failure for a completely missing object.

        Args:
            result: Result to add failure to.
            context: Evaluation context.
            obj_name: Name of missing object.
            obj_class: Class of missing object.
            reason: Why the object is considered missing.
        """
        frame_path = context.get_frame_path(0)
        result.add_failure(
            timestamp=context.frame_to_timestamp(0),
            frame_number=0,
            failure_type="object_missing",
            severity="critical",
            message=f"Object '{obj_name}' (class: {obj_class}) not found: {reason}",
            object=obj_name,
            suggested_fix="Ensure the object is present and clearly visible in the video"
        )
        result.score = 0.0
        result.passed = False

        result.add_evidence(
            timestamp=context.frame_to_timestamp(0),
            frame_number=0,
            thumbnail_path=str(context.get_thumbnail_path(0, "missing_object")),
            explanation=f"Object '{obj_name}' was not detected in any frame",
            confidence=0.0
        )
