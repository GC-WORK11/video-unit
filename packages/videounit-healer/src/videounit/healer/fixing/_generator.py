"""Fix proposal generators for VideoUnit Self-Healing Pipeline."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from videounit.healer._base import (
    DiagnosisResult,
    FixGenerator,
    FixProposal,
    RepairStrategy,
)

if TYPE_CHECKING:
    from videounit_evaluators._result import Failure


class PromptEnhancementGenerator(FixGenerator):
    """Generator for prompt enhancement fixes.

    Creates fixes that improve the clarity, specificity, and completeness
    of test prompts to reduce ambiguity and improve VLM interpretation.
    """

    strategy = RepairStrategy.PROMPT_ENHANCEMENT

    async def generate(
        self, diagnosis: DiagnosisResult, contract: dict[str, Any]
    ) -> FixProposal:
        """Generate a prompt enhancement fix.

        Args:
            diagnosis: The diagnosis to generate a fix for.
            contract: The current contract with prompt and assertions.

        Returns:
            A FixProposal with prompt modifications.
        """
        root_cause = diagnosis.root_cause
        evidence = diagnosis.evidence

        prompt_delta = ""
        risks: list[str] = []

        original_prompt = contract.get("prompt", "")
        evidence_analysis = evidence.get("prompt_verbosity", "moderate")

        if root_cause == "high_prompt_ambiguity":
            ambiguity_indicators = evidence.get("ambiguity_indicators", [])
            prompt_delta = self._enhance_specificity(
                original_prompt, ambiguity_indicators
            )
            risks.append("Enhanced prompt may be longer and more complex")

        elif root_cause == "prompt_contains_ambiguous_terms":
            prompt_delta = self._replace_ambiguous_terms(original_prompt)
            risks.append("Revised terms may change original intent slightly")

        elif root_cause == "low_detection_confidence":
            target_object = evidence.get("target_object", "")
            prompt_delta = self._add_object_context(original_prompt, target_object)
            risks.append("Added context may bias VLM interpretation")

        elif root_cause in ("object_not_detected_in_frames", "object_absence_or_occlusion"):
            target_object = evidence.get("expected_object", evidence.get("target_object", ""))
            prompt_delta = self._clarify_object_expectation(original_prompt, target_object)
            risks.append("Clarified expectation may increase strictness")

        elif root_cause == "color_mismatch_unquantified":
            prompt_delta = self._quantify_color_expectations(original_prompt)
            risks.append("Quantified colors may reduce false positives but increase false negatives")

        elif root_cause == "significant_color_deviation":
            prompt_delta = self._add_color_tolerance_guidance(original_prompt)
            risks.append("Tolerance guidance may affect strictness of color checks")

        else:
            prompt_delta = self._general_prompt_improvement(original_prompt)
            risks.append("General improvements may have unforeseen effects")

        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            target_failure_type=root_cause,
            prompt_delta=prompt_delta,
            contract_modifications={},
            confidence=0.82,
            risks=risks,
        )

    def _enhance_specificity(
        self, prompt: str, ambiguity_indicators: list[str]
    ) -> str:
        """Enhance prompt specificity based on ambiguity indicators.

        Args:
            prompt: Original prompt text.
            ambiguity_indicators: List of ambiguous terms or phrases found.

        Returns:
            Enhanced prompt with improved specificity.
        """
        enhanced = prompt

        if len(prompt.split()) < 10:
            enhanced = enhanced + (
                " The object should remain visible throughout the entire video sequence "
                "without sudden disappearances or occlusions."
            )

        for indicator in ambiguity_indicators:
            if indicator == "vague_quantity":
                enhanced = enhanced.replace(
                    "several objects", "at least 3 distinct objects"
                )
            elif indicator == "undefined_color":
                enhanced = enhanced + " Colors should match the specified RGB values within a delta_e of 10."

        return enhanced

    def _replace_ambiguous_terms(self, prompt: str) -> str:
        """Replace ambiguous terms with precise alternatives.

        Args:
            prompt: Original prompt text.

        Returns:
            Prompt with ambiguous terms replaced.
        """
        replacements = {
            "near": "within 10 pixels of",
            "close": "adjacent to or within 20 pixels of",
            "far": "more than 100 pixels from",
            "quickly": "in less than 0.5 seconds",
            "slowly": "over more than 2 seconds",
            "many": "more than 5",
            "few": "fewer than 3",
            "bright": "with luminance value above 200 (on 0-255 scale)",
            "dark": "with luminance value below 50 (on 0-255 scale)",
        }

        result = prompt
        for ambiguous, precise in replacements.items():
            result = result.replace(ambiguous, precise)

        return result

    def _add_object_context(
        self, prompt: str, target_object: str
    ) -> str:
        """Add context about expected object to prompt.

        Args:
            prompt: Original prompt text.
            target_object: The object being detected.

        Returns:
            Prompt with additional object context.
        """
        return (
            f"{prompt} "
            f"Pay special attention to {target_object} - if present, it should have "
            f"clear visual features and be distinguishable from background elements."
        )

    def _clarify_object_expectation(
        self, prompt: str, target_object: str
    ) -> str:
        """Clarify expectations about object visibility.

        Args:
            prompt: Original prompt text.
            target_object: The expected object.

        Returns:
            Prompt with clarified object expectations.
        """
        return (
            f"{prompt} "
            f"Expected behavior: {target_object} should be clearly visible "
            f"throughout the video. If {target_object} is not detected in a frame, "
            f"this indicates a failure unless it is explicitly occluded by another object."
        )

    def _quantify_color_expectations(self, prompt: str) -> str:
        """Add quantitative color specifications to prompt.

        Args:
            prompt: Original prompt text.

        Returns:
            Prompt with quantified color expectations.
        """
        return (
            f"{prompt} "
            f"Color matching should use CIEDE2000 color difference metric "
            f"with a tolerance of delta_e <= 15 for close matches and "
            f"delta_e <= 30 for acceptable matches."
        )

    def _add_color_tolerance_guidance(self, prompt: str) -> str:
        """Add color tolerance guidance to prompt.

        Args:
            prompt: Original prompt text.

        Returns:
            Prompt with color tolerance guidance.
        """
        return (
            f"{prompt} "
            f"For color comparisons, allow for natural video compression artifacts. "
            f"Ignore differences in saturation caused by encoding quality variations."
        )

    def _general_prompt_improvement(self, prompt: str) -> str:
        """Apply general prompt improvements.

        Args:
            prompt: Original prompt text.

        Returns:
            Improved prompt.
        """
        if not prompt.endswith("."):
            prompt = prompt + "."

        return (
            f"{prompt} "
            f"Ensure all assertions are verifiable against frame-by-frame analysis. "
            f"Use precise spatial relationships (e.g., 'directly above' not 'on top of')."
        )


class ToleranceAdjustmentGenerator(FixGenerator):
    """Generator for tolerance adjustment fixes.

    Creates fixes that adjust tolerance thresholds in assertions to
    account for natural video variation while maintaining test validity.
    """

    strategy = RepairStrategy.TOLERANCE_ADJUSTMENT

    async def generate(
        self, diagnosis: DiagnosisResult, contract: dict[str, Any]
    ) -> FixProposal:
        """Generate a tolerance adjustment fix.

        Args:
            diagnosis: The diagnosis to generate a fix for.
            contract: The current contract with assertions.

        Returns:
            A FixProposal with tolerance modifications.
        """
        root_cause = diagnosis.root_cause
        evidence = diagnosis.evidence

        contract_modifications: dict[str, Any] = {}
        risks: list[str] = []

        assertions = contract.get("assertions", [])

        if root_cause == "moderate_color_shift":
            contract_modifications = self._relax_color_tolerance(assertions)
            risks.append("Relaxed tolerance may miss genuine color issues")

        elif root_cause == "minor_temporal_fluctuation":
            contract_modifications = self._relax_temporal_tolerance(assertions)
            risks.append("May reduce sensitivity to subtle temporal issues")

        elif root_cause == "moderate_temporal_instability":
            contract_modifications = self._adjust_temporal_windows(assertions)
            risks.append("Adjusted windows may allow some flickering")

        elif root_cause == "intermittent_temporal_artifact":
            contract_modifications = self._add_artifact_filtering(assertions)
            risks.append("Filtering may mask real intermittent failures")

        elif root_cause == "object_absence_or_occlusion":
            contract_modifications = self._add_occlusion_handling(assertions)
            risks.append("Occlusion handling may miss actual object tracking failures")

        elif root_cause == "high_temporal_variance":
            contract_modifications = self._add_moving_average_smoothing(assertions)
            risks.append("Smoothing may hide frame-level issues")

        else:
            contract_modifications = self._relax_general_tolerance(assertions)
            risks.append("General tolerance relaxation may reduce test strictness")

        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            target_failure_type=root_cause,
            prompt_delta="",
            contract_modifications=contract_modifications,
            confidence=0.78,
            risks=risks,
        )

    def _relax_color_tolerance(self, assertions: list[dict[str, Any]]) -> dict[str, Any]:
        """Relax color tolerance in assertions.

        Args:
            assertions: Current assertion list.

        Returns:
            Contract modifications for color tolerance.
        """
        return {
            "assertion_adjustments": [
                {
                    "type": "color_tolerance",
                    "adjustment": "relax",
                    "delta_e_increase": 5,
                    "affected_assertions": [
                        a.get("id") for a in assertions if a.get("assertion_type") == "color_match"
                    ],
                }
            ]
        }

    def _relax_temporal_tolerance(self, assertions: list[dict[str, Any]]) -> dict[str, Any]:
        """Relax temporal tolerance in assertions.

        Args:
            assertions: Current assertion list.

        Returns:
            Contract modifications for temporal tolerance.
        """
        return {
            "assertion_adjustments": [
                {
                    "type": "temporal_tolerance",
                    "adjustment": "relax",
                    "frame_window_increase": 2,
                    "affected_assertions": [
                        a.get("id") for a in assertions if a.get("assertion_type") == "temporal"
                    ],
                }
            ]
        }

    def _adjust_temporal_windows(self, assertions: list[dict[str, Any]]) -> dict[str, Any]:
        """Adjust temporal analysis windows.

        Args:
            assertions: Current assertion list.

        Returns:
            Contract modifications for temporal windows.
        """
        return {
            "assertion_adjustments": [
                {
                    "type": "temporal_window",
                    "adjustment": "expand",
                    "window_size_multiplier": 1.5,
                    "affected_assertions": [
                        a.get("id") for a in assertions if "temporal" in a.get("assertion_type", "")
                    ],
                }
            ]
        }

    def _add_artifact_filtering(self, assertions: list[dict[str, Any]]) -> dict[str, Any]:
        """Add artifact filtering to reduce false positives.

        Args:
            assertions: Current assertion list.

        Returns:
            Contract modifications for artifact filtering.
        """
        return {
            "assertion_adjustments": [
                {
                    "type": "artifact_filter",
                    "adjustment": "add",
                    "filter_config": {
                        "min_duration_frames": 3,
                        "ignore_single_frame_anomalies": True,
                    },
                    "affected_assertions": [
                        a.get("id") for a in assertions if "flicker" in a.get("assertion_type", "")
                    ],
                }
            ]
        }

    def _add_occlusion_handling(self, assertions: list[dict[str, Any]]) -> dict[str, Any]:
        """Add occlusion handling to object detection.

        Args:
            assertions: Current assertion list.

        Returns:
            Contract modifications for occlusion handling.
        """
        return {
            "assertion_adjustments": [
                {
                    "type": "occlusion_handling",
                    "adjustment": "add",
                    "config": {
                        "allow_partial_occlusion": True,
                        "min_visibility_threshold": 0.3,
                        "recovery_frames": 5,
                    },
                    "affected_assertions": [
                        a.get("id") for a in assertions if "object" in a.get("assertion_type", "")
                    ],
                }
            ]
        }

    def _add_moving_average_smoothing(self, assertions: list[dict[str, Any]]) -> dict[str, Any]:
        """Add moving average smoothing for temporal analysis.

        Args:
            assertions: Current assertion list.

        Returns:
            Contract modifications for smoothing.
        """
        return {
            "assertion_adjustments": [
                {
                    "type": "temporal_smoothing",
                    "adjustment": "add",
                    "window_size": 5,
                    "affected_assertions": [
                        a.get("id") for a in assertions if "stability" in a.get("assertion_type", "")
                    ],
                }
            ]
        }

    def _relax_general_tolerance(self, assertions: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply general tolerance relaxation.

        Args:
            assertions: Current assertion list.

        Returns:
            Contract modifications for general tolerance.
        """
        return {
            "assertion_adjustments": [
                {
                    "type": "general_tolerance",
                    "adjustment": "relax",
                    "relaxation_factor": 1.2,
                    "affected_assertions": [a.get("id") for a in assertions],
                }
            ]
        }


class AssertionAdditionGenerator(FixGenerator):
    """Generator for assertion addition fixes.

    Creates fixes that add new assertions to the contract to catch
    edge cases that were previously untested.
    """

    strategy = RepairStrategy.ASSERTION_ADDITION

    async def generate(
        self, diagnosis: DiagnosisResult, contract: dict[str, Any]
    ) -> FixProposal:
        """Generate an assertion addition fix.

        Args:
            diagnosis: The diagnosis to generate a fix for.
            contract: The current contract with assertions.

        Returns:
            A FixProposal with new assertions to add.
        """
        root_cause = diagnosis.root_cause

        new_assertions: list[dict[str, Any]] = []
        risks: list[str] = []

        if root_cause == "minor_temporal_fluctuation":
            new_assertions.append(
                {
                    "assertion_type": "temporal_consistency",
                    "description": "Frame-to-frame differences should not exceed threshold",
                    "parameters": {
                        "max_frame_difference": 0.15,
                        "metric": "ssim",
                    },
                }
            )
            risks.append("New assertion may increase test sensitivity")

        elif root_cause == "color_variation_not_tested":
            new_assertions.append(
                {
                    "assertion_type": "color_consistency",
                    "description": "Colors should remain consistent across frames",
                    "parameters": {
                        "max_color_variance": 5.0,
                        "sample_rate": 10,
                    },
                }
            )
            risks.append("May flag natural video compression artifacts as failures")

        elif root_cause == "object_tracking_gap":
            new_assertions.append(
                {
                    "assertion_type": "object_continuity",
                    "description": "Objects should be trackable across all frames",
                    "parameters": {
                        "max_gap_frames": 3,
                        "min_track_length": 10,
                    },
                }
            )
            risks.append("May fail on legitimate occlusions")

        else:
            new_assertions.append(
                {
                    "assertion_type": "general_robustness",
                    "description": "Video should maintain consistent quality throughout",
                    "parameters": {
                        "min_quality_score": 0.7,
                    },
                }
            )
            risks.append("General assertions may be too broad")

        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            target_failure_type=root_cause,
            prompt_delta="",
            contract_modifications={
                "add_assertions": new_assertions,
            },
            confidence=0.72,
            risks=risks,
        )


class FullRegenerationGenerator(FixGenerator):
    """Generator for full regeneration fixes.

    Creates fixes that require complete regeneration of the video or
    major changes to the test setup, used for severe failures.
    """

    strategy = RepairStrategy.FULL_REGENERATION

    async def generate(
        self, diagnosis: DiagnosisResult, contract: dict[str, Any]
    ) -> FixProposal:
        """Generate a full regeneration fix.

        Args:
            diagnosis: The diagnosis to generate a fix for.
            contract: The current contract.

        Returns:
            A FixProposal requiring full regeneration.
        """
        root_cause = diagnosis.root_cause
        evidence = diagnosis.evidence

        prompt_delta = ""
        contract_modifications: dict[str, Any] = {}
        risks: list[str] = []

        if root_cause == "excessive_flickering":
            prompt_delta = (
                "CRITICAL: Regenerate video with stable frame rates. "
                "No flickering or frame drops allowed. "
                "Use consistent 30fps or 60fps throughout."
            )
            contract_modifications = {
                "add_assertions": [
                    {
                        "assertion_type": "frame_stability",
                        "description": "No frame rate fluctuations or flickering",
                        "parameters": {"max_flicker_cycles": 2},
                    }
                ]
            }
            risks.append("May require significant regeneration effort")

        elif root_cause in ("invalid_gravity_simulation", "invalid_momentum_conservation"):
            prompt_delta = (
                "CRITICAL: Physics simulation is invalid. "
                "Regenerate ensuring proper physical laws: "
                "gravity at 9.8 m/s^2, momentum conservation, "
                "no impossible accelerations or sudden stops."
            )
            contract_modifications = {
                "add_assertions": [
                    {
                        "assertion_type": "physics_sanity",
                        "description": "Physics simulation must be valid",
                        "parameters": {"strict_mode": True},
                    }
                ]
            }
            risks.append("May require complete scene recreation")

        elif root_cause == "collision_detection_failure":
            prompt_delta = (
                "CRITICAL: Objects are penetrating each other. "
                "Regenerate with proper collision detection. "
                "Objects should bounce or stop at contact, not clip through."
            )
            contract_modifications = {
                "add_assertions": [
                    {
                        "assertion_type": "collision_integrity",
                        "description": "Objects must not penetrate each other",
                        "parameters": {"max_penetration_depth": 0.01},
                    }
                ]
            }
            risks.append("Physics engine may need adjustment")

        elif root_cause == "kinematics_violation":
            prompt_delta = (
                "CRITICAL: Motion path is physically impossible. "
                "Regenerate with valid kinematics: "
                "continuous paths, realistic velocities, no teleporting."
            )
            contract_modifications = {
                "add_assertions": [
                    {
                        "assertion_type": "motion_continuity",
                        "description": "Motion must follow valid kinematic paths",
                        "parameters": {"max_velocity_change": 10.0},
                    }
                ]
            }
            risks.append("Animation system may need recalibration")

        elif root_cause == "high_temporal_variance":
            prompt_delta = (
                "CRITICAL: Video has extreme frame-to-frame variation. "
                "Regenerate with consistent encoding settings, "
                "stable bitrate, and proper frame ordering."
            )
            contract_modifications = {
                "add_assertions": [
                    {
                        "assertion_type": "encoding_stability",
                        "description": "Encoding must be consistent throughout",
                        "parameters": {"max_bitrate_variance": 0.2},
                    }
                ]
            }
            risks.append("May require different encoding pipeline")

        else:
            prompt_delta = (
                "CRITICAL: Fundamental issue detected requiring regeneration. "
                "Review the entire video generation process and regenerate "
                "ensuring all quality metrics are met."
            )
            risks.append("Unknown root cause - regeneration may not guarantee fix")

        return FixProposal(
            proposal_id=str(uuid.uuid4()),
            target_failure_type=root_cause,
            prompt_delta=prompt_delta,
            contract_modifications=contract_modifications,
            confidence=0.65,
            risks=risks,
        )


def create_default_fix_generators() -> list[FixGenerator]:
    """Create list of all default fix generators.

    Returns:
        List of generator instances for all repair strategies.
    """
    return [
        PromptEnhancementGenerator(),
        ToleranceAdjustmentGenerator(),
        AssertionAdditionGenerator(),
        FullRegenerationGenerator(),
    ]
