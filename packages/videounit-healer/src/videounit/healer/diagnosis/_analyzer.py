"""Diagnosis analyzers for VideoUnit Self-Healing Pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from videounit.healer._base import (
    DiagnosisResult,
    FailureAnalyzer,
    RepairStrategy,
)

if TYPE_CHECKING:
    from videounit_evaluators._result import Failure


class ObjectDetectionAnalyzer(FailureAnalyzer):
    """Analyzer for object detection related failures.

    Handles failures where objects are missing, misidentified, or have
    incorrect bounding boxes.
    """

    failure_type = "object_detection"

    async def analyze(
        self, failure: Failure, context: dict[str, Any]
    ) -> DiagnosisResult:
        """Analyze an object detection failure.

        Args:
            failure: The failure to analyze.
            context: Context including frame analysis, detection confidence scores.

        Returns:
            DiagnosisResult with root cause and evidence.
        """
        evidence: dict[str, Any] = {
            "failure_message": failure.message,
            "failure_type": failure.type,
        }

        if failure.object:
            evidence["target_object"] = failure.object

        frame_analysis = context.get("frame_analysis", {})
        detection_scores = context.get("detection_scores", {})

        if failure.object and failure.object in detection_scores:
            obj_scores = detection_scores[failure.object]
            avg_confidence = sum(obj_scores) / len(obj_scores) if obj_scores else 0.0
            evidence["average_confidence"] = avg_confidence

            if avg_confidence < 0.5:
                root_cause = "low_detection_confidence"
                strategy = RepairStrategy.PROMPT_ENHANCEMENT
                evidence["threshold_used"] = 0.5
            else:
                root_cause = "object_absence_or_occlusion"
                strategy = RepairStrategy.TOLERANCE_ADJUSTMENT
        else:
            root_cause = "object_not_detected_in_frames"
            strategy = RepairStrategy.PROMPT_ENHANCEMENT
            evidence["expected_object"] = failure.object

        return DiagnosisResult(
            root_cause=root_cause,
            evidence=evidence,
            confidence=0.85,
            repair_strategy=strategy,
        )

    def can_handle(self, failure: Failure) -> bool:
        """Check if this analyzer handles the failure.

        Args:
            failure: The failure to check.

        Returns:
            True if the failure is related to object detection.
        """
        object_failure_types = {
            "object_missing",
            "object_not_detected",
            "detection_confidence_low",
            "bbox_incorrect",
            "object_occluded",
        }
        return failure.type in object_failure_types or "object" in failure.type.lower()


class ColorMismatchAnalyzer(FailureAnalyzer):
    """Analyzer for color-related failures.

    Handles failures where colors are incorrect, inconsistent, or do not
    match the expected values in the prompt.
    """

    failure_type = "color"

    async def analyze(
        self, failure: Failure, context: dict[str, Any]
    ) -> DiagnosisResult:
        """Analyze a color mismatch failure.

        Args:
            failure: The failure to analyze.
            context: Context including color histogram data, expected vs actual colors.

        Returns:
            DiagnosisResult with root cause and evidence.
        """
        evidence: dict[str, Any] = {
            "failure_message": failure.message,
        }

        color_data = context.get("color_analysis", {})
        expected_colors = context.get("expected_colors", {})
        actual_colors = context.get("actual_colors", {})

        if expected_colors and actual_colors:
            evidence["expected_colors"] = expected_colors
            evidence["actual_colors"] = actual_colors

            delta_e_values = color_data.get("color_delta_e", [])
            if delta_e_values:
                avg_delta_e = sum(delta_e_values) / len(delta_e_values)
                evidence["average_delta_e"] = avg_delta_e

                if avg_delta_e > 30:
                    root_cause = "significant_color_deviation"
                    strategy = RepairStrategy.PROMPT_ENHANCEMENT
                elif avg_delta_e > 15:
                    root_cause = "moderate_color_shift"
                    strategy = RepairStrategy.TOLERANCE_ADJUSTMENT
                else:
                    root_cause = "minor_color_variation"
                    strategy = RepairStrategy.TOLERANCE_ADJUSTMENT
            else:
                root_cause = "color_mismatch_unquantified"
                strategy = RepairStrategy.PROMPT_ENHANCEMENT
        else:
            root_cause = "color_comparison_failed"
            strategy = RepairStrategy.PROMPT_ENHANCEMENT

        return DiagnosisResult(
            root_cause=root_cause,
            evidence=evidence,
            confidence=0.80,
            repair_strategy=strategy,
        )

    def can_handle(self, failure: Failure) -> bool:
        """Check if this analyzer handles the failure.

        Args:
            failure: The failure to check.

        Returns:
            True if the failure is color-related.
        """
        color_failure_types = {
            "color_mismatch",
            "color_shift",
            "wrong_color",
            "color_inconsistent",
            "color_deviation",
        }
        return failure.type in color_failure_types or "color" in failure.type.lower()


class TemporalInstabilityAnalyzer(FailureAnalyzer):
    """Analyzer for temporal stability failures.

    Handles failures related to flickering, jittering, or inconsistent
    behavior over time in the video.
    """

    failure_type = "temporal"

    async def analyze(
        self, failure: Failure, context: dict[str, Any]
    ) -> DiagnosisResult:
        """Analyze a temporal instability failure.

        Args:
            failure: The failure to analyze.
            context: Context including frame-to-frame metrics, temporal scores.

        Returns:
            DiagnosisResult with root cause and evidence.
        """
        evidence: dict[str, Any] = {
            "failure_message": failure.message,
        }

        temporal_data = context.get("temporal_analysis", {})
        frame_scores = context.get("frame_stability_scores", [])

        if frame_scores:
            variance = self._calculate_variance(frame_scores)
            evidence["stability_variance"] = variance
            evidence["num_frames_analyzed"] = len(frame_scores)

            if variance > 0.3:
                root_cause = "high_temporal_variance"
                strategy = RepairStrategy.FULL_REGENERATION
            elif variance > 0.1:
                root_cause = "moderate_temporal_instability"
                strategy = RepairStrategy.TOLERANCE_ADJUSTMENT
            else:
                root_cause = "minor_temporal_fluctuation"
                strategy = RepairStrategy.ASSERTION_ADDITION

            evidence["stability_threshold"] = 0.1
        else:
            flicker_cycles = temporal_data.get("flicker_cycles", 0)
            evidence["flicker_cycles_detected"] = flicker_cycles

            if flicker_cycles > 5:
                root_cause = "excessive_flickering"
                strategy = RepairStrategy.FULL_REGENERATION
            else:
                root_cause = "intermittent_temporal_artifact"
                strategy = RepairStrategy.TOLERANCE_ADJUSTMENT

        return DiagnosisResult(
            root_cause=root_cause,
            evidence=evidence,
            confidence=0.78,
            repair_strategy=strategy,
        )

    @staticmethod
    def _calculate_variance(values: list[float]) -> float:
        """Calculate variance of a list of values.

        Args:
            values: List of numeric values.

        Returns:
            The variance of the values.
        """
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        squared_diffs = [(v - mean) ** 2 for v in values]
        return sum(squared_diffs) / len(squared_diffs)

    def can_handle(self, failure: Failure) -> bool:
        """Check if this analyzer handles the failure.

        Args:
            failure: The failure to check.

        Returns:
            True if the failure is temporal-related.
        """
        temporal_failure_types = {
            "temporal_flicker",
            "jitter",
            "temporal_instability",
            "frame_inconsistency",
            "motion_jitter",
            "flickering",
        }
        return failure.type in temporal_failure_types or "temporal" in failure.type.lower()


class PromptAmbiguityAnalyzer(FailureAnalyzer):
    """Analyzer for prompt-related failures.

    Handles failures where the test prompt is ambiguous, unclear, or
    leads to inconsistent interpretation by the VLM.
    """

    failure_type = "prompt"

    async def analyze(
        self, failure: Failure, context: dict[str, Any]
    ) -> DiagnosisResult:
        """Analyze a prompt ambiguity failure.

        Args:
            failure: The failure to analyze.
            context: Context including prompt text, VLM interpretation results.

        Returns:
            DiagnosisResult with root cause and evidence.
        """
        evidence: dict[str, Any] = {
            "failure_message": failure.message,
        }

        prompt_data = context.get("prompt_analysis", {})
        original_prompt = context.get("original_prompt", "")
        vlm_interpretations = context.get("vlm_interpretations", [])

        if original_prompt:
            evidence["original_prompt_length"] = len(original_prompt)
            evidence["word_count"] = len(original_prompt.split())

            if len(original_prompt.split()) < 5:
                evidence["prompt_verbosity"] = "minimal"
            elif len(original_prompt.split()) > 50:
                evidence["prompt_verbosity"] = "verbose"
            else:
                evidence["prompt_verbosity"] = "moderate"

        if vlm_interpretations:
            unique_interpretations = len(set(vlm_interpretations))
            evidence["unique_interpretations"] = unique_interpretations
            evidence["total_interpretations"] = len(vlm_interpretations)

            if unique_interpretations > len(vlm_interpretations) * 0.5:
                root_cause = "high_prompt_ambiguity"
                strategy = RepairStrategy.PROMPT_ENHANCEMENT
            else:
                root_cause = "minor_prompt_unclarity"
                strategy = RepairStrategy.PROMPT_ENHANCEMENT
        else:
            ambiguity_indicators = prompt_data.get("ambiguity_indicators", [])
            evidence["ambiguity_indicators"] = ambiguity_indicators

            if len(ambiguity_indicators) > 2:
                root_cause = "prompt_contains_ambiguous_terms"
                strategy = RepairStrategy.PROMPT_ENHANCEMENT
            else:
                root_cause = "prompt_interpretation_variance"
                strategy = RepairStrategy.PROMPT_ENHANCEMENT

        return DiagnosisResult(
            root_cause=root_cause,
            evidence=evidence,
            confidence=0.75,
            repair_strategy=strategy,
        )

    def can_handle(self, failure: Failure) -> bool:
        """Check if this analyzer handles the failure.

        Args:
            failure: The failure to check.

        Returns:
            True if the failure is prompt-related.
        """
        prompt_failure_types = {
            "prompt_ambiguous",
            "prompt_unclear",
            "vlm_misinterpretation",
            "assertion_vague",
            "criteria_not_specific",
        }
        return failure.type in prompt_failure_types or "prompt" in failure.type.lower()


class PhysicsViolationAnalyzer(FailureAnalyzer):
    """Analyzer for physics-related failures.

    Handles failures where the video violates physical laws such as
    gravity, momentum, or object permanence.
    """

    failure_type = "physics"

    async def analyze(
        self, failure: Failure, context: dict[str, Any]
    ) -> DiagnosisResult:
        """Analyze a physics violation failure.

        Args:
            failure: The failure to analyze.
            context: Context including physics simulation data, violation details.

        Returns:
            DiagnosisResult with root cause and evidence.
        """
        evidence: dict[str, Any] = {
            "failure_message": failure.message,
        }

        physics_data = context.get("physics_analysis", {})
        violation_type = physics_data.get("violation_type", failure.type)
        severity = physics_data.get("severity", "unknown")

        evidence["violation_type"] = violation_type
        evidence["reported_severity"] = severity

        if violation_type in ("gravity_violation", "free_fall_impossible"):
            root_cause = "invalid_gravity_simulation"
            strategy = RepairStrategy.FULL_REGENERATION
        elif violation_type in ("momentum_violation", "instant_stop"):
            root_cause = "invalid_momentum_conservation"
            strategy = RepairStrategy.FULL_REGENERATION
        elif violation_type in ("object_penetration", "clipping"):
            root_cause = "collision_detection_failure"
            strategy = RepairStrategy.FULL_REGENERATION
        elif violation_type == "impossible_motion":
            root_cause = "kinematics_violation"
            strategy = RepairStrategy.FULL_REGENERATION
        else:
            root_cause = "physics_simulation_error"
            strategy = RepairStrategy.FULL_REGENERATION

        return DiagnosisResult(
            root_cause=root_cause,
            evidence=evidence,
            confidence=0.90,
            repair_strategy=strategy,
        )

    def can_handle(self, failure: Failure) -> bool:
        """Check if this analyzer handles the failure.

        Args:
            failure: The failure to check.

        Returns:
            True if the failure is physics-related.
        """
        physics_failure_types = {
            "physics_violation",
            "gravity_violation",
            "momentum_violation",
            "collision_failure",
            "object_penetration",
            "impossible_motion",
            "physics_sanity",
        }
        return failure.type in physics_failure_types or "physics" in failure.type.lower()


def create_default_diagnosis_engine() -> DiagnosisEngine:
    """Create a diagnosis engine with all default analyzers registered.

    Returns:
        A DiagnosisEngine with all standard analyzers registered.
    """
    from videounit.healer._base import DiagnosisEngine

    engine = DiagnosisEngine()
    engine.register(ObjectDetectionAnalyzer())
    engine.register(ColorMismatchAnalyzer())
    engine.register(TemporalInstabilityAnalyzer())
    engine.register(PromptAmbiguityAnalyzer())
    engine.register(PhysicsViolationAnalyzer())
    return engine
