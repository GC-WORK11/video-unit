"""Self-Healing Pipeline for VideoUnit AI Video Testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from videounit.healer._base import (
    DiagnosisEngine,
    DiagnosisResult,
    FixProposal,
    FixProposalGenerator,
    RepairStrategy,
    SelfHealingReport,
)
from videounit.healer.diagnosis import create_default_diagnosis_engine
from videounit.healer.fixing import create_default_fix_generators

if TYPE_CHECKING:
    from videounit_evaluators._result import EvaluationResult, Failure


@dataclass
class PipelineConfig:
    """Configuration for the self-healing pipeline.

    Attributes:
        max_iterations: Maximum number of healing iterations to perform.
        convergence_threshold: Number of consecutive passes to consider
            the pipeline converged.
        apply_fixes_automatically: Whether to automatically apply fixes
            or just generate proposals.
        confidence_threshold: Minimum confidence required to apply a fix
            automatically (0.0 to 1.0).
    """

    max_iterations: int = 3
    convergence_threshold: int = 1
    apply_fixes_automatically: bool = False
    confidence_threshold: float = 0.75


class SelfHealingPipeline:
    """Self-healing pipeline that diagnoses failures and generates fixes.

    The pipeline orchestrates the diagnosis engine and fix proposal generator
    to analyze evaluation failures and propose or apply fixes to contracts.

    Attributes:
        config: Pipeline configuration.
        diagnosis_engine: Engine for diagnosing failures.
        fix_generator: Generator for creating fix proposals.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        diagnosis_engine: DiagnosisEngine | None = None,
        fix_generator: FixProposalGenerator | None = None,
    ) -> None:
        """Initialize the self-healing pipeline.

        Args:
            config: Optional pipeline configuration. Uses defaults if not provided.
            diagnosis_engine: Optional diagnosis engine. Creates default if not provided.
            fix_generator: Optional fix generator. Creates default if not provided.
        """
        self.config = config or PipelineConfig()
        self.diagnosis_engine = diagnosis_engine or create_default_diagnosis_engine()
        self.fix_generator = fix_generator or FixProposalGenerator()

        for generator in create_default_fix_generators():
            self.fix_generator.register(generator)

    async def heal(
        self,
        evaluation_result: EvaluationResult,
        contract: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> SelfHealingReport:
        """Run the self-healing pipeline on evaluation results.

        Args:
            evaluation_result: The evaluation result containing failures.
            contract: The contract that was evaluated.
            context: Optional additional context for diagnosis.

        Returns:
            A SelfHealingReport summarizing the healing process.
        """
        if context is None:
            context = {}

        failures = evaluation_result.failures
        total_failures = len(failures)

        report = SelfHealingReport(total_failures=total_failures)

        if total_failures == 0:
            report.converged = True
            return report

        all_failures = self._collect_all_failures(failures)
        diagnoses = await self._diagnose_failures(all_failures, context)
        fix_proposals = await self._generate_fixes(diagnoses, contract)

        report.diagnoses = diagnoses
        report.fix_proposals = fix_proposals

        current_iteration = 0
        consecutive_passes = 0

        while current_iteration < self.config.max_iterations:
            current_iteration += 1
            report.iterations = current_iteration

            applied_proposals = await self._apply_fixes(
                fix_proposals, contract, evaluation_result, context
            )
            report.applied_fixes.extend(applied_proposals)

            if not applied_proposals:
                break

            new_failures = await self._reevaluate(contract, context)
            new_diagnoses = await self._diagnose_failures(new_failures, context)

            if not new_failures:
                consecutive_passes += 1
                if consecutive_passes >= self.config.convergence_threshold:
                    report.converged = True
                    break
            else:
                consecutive_passes = 0

            diagnoses = new_diagnoses
            new_proposals = await self._generate_fixes(new_diagnoses, contract)
            fix_proposals = new_proposals

        return report

    def _collect_all_failures(
        self, failures: list[Failure]
    ) -> list[Failure]:
        """Collect all failures into a flat list.

        Args:
            failures: List of failures to collect.

        Returns:
            Flat list of all failures.
        """
        return list(failures)

    async def _diagnose_failures(
        self,
        failures: list[Failure],
        context: dict[str, Any],
    ) -> list[DiagnosisResult]:
        """Diagnose all failures.

        Args:
            failures: Failures to diagnose.
            context: Context for diagnosis.

        Returns:
            List of diagnosis results.
        """
        diagnoses = []
        for failure in failures:
            diagnosis = await self.diagnosis_engine.diagnose(failure, context)
            diagnoses.append(diagnosis)
        return diagnoses

    async def _generate_fixes(
        self,
        diagnoses: list[DiagnosisResult],
        contract: dict[str, Any],
    ) -> list[FixProposal]:
        """Generate fix proposals for all diagnoses.

        Args:
            diagnoses: Diagnoses to generate fixes for.
            contract: Current contract.

        Returns:
            List of fix proposals.
        """
        return await self.fix_generator.generate_batch(diagnoses, contract)

    async def _apply_fixes(
        self,
        proposals: list[FixProposal],
        contract: dict[str, Any],
        evaluation_result: EvaluationResult,
        context: dict[str, Any],
    ) -> list[FixProposal]:
        """Apply fixes based on proposals and configuration.

        Args:
            proposals: Fix proposals to potentially apply.
            contract: Current contract (will be modified if fixes applied).
            evaluation_result: Current evaluation result.
            context: Context for evaluation.

        Returns:
            List of successfully applied fix proposals.
        """
        applied: list[FixProposal] = []

        if not self.config.apply_fixes_automatically:
            return applied

        for proposal in proposals:
            if proposal.confidence < self.config.confidence_threshold:
                continue

            if self._is_safe_to_apply(proposal, contract):
                self._apply_proposal_to_contract(proposal, contract)
                applied.append(proposal)

        return applied

    def _is_safe_to_apply(
        self,
        proposal: FixProposal,
        contract: dict[str, Any],
    ) -> bool:
        """Check if a proposal is safe to apply.

        Args:
            proposal: The fix proposal to check.
            contract: The current contract.

        Returns:
            True if the proposal is safe to apply.
        """
        high_risk_strategies = {RepairStrategy.FULL_REGENERATION}
        if proposal.confidence < 0.5:
            return False

        if proposal.repair_strategy in high_risk_strategies and proposal.confidence < 0.8:
            return False

        return True

    def _apply_proposal_to_contract(
        self,
        proposal: FixProposal,
        contract: dict[str, Any],
    ) -> None:
        """Apply a fix proposal to the contract.

        Args:
            proposal: The proposal to apply.
            contract: The contract to modify.
        """
        if proposal.prompt_delta:
            current_prompt = contract.get("prompt", "")
            contract["prompt"] = current_prompt + " " + proposal.prompt_delta

        if proposal.contract_modifications:
            if "add_assertions" in proposal.contract_modifications:
                existing = contract.get("assertions", [])
                new_assertions = proposal.contract_modifications["add_assertions"]
                contract["assertions"] = existing + new_assertions

            if "assertion_adjustments" in proposal.contract_modifications:
                adjustments = proposal.contract_modifications["assertion_adjustments"]
                self._apply_assertion_adjustments(contract, adjustments)

    def _apply_assertion_adjustments(
        self,
        contract: dict[str, Any],
        adjustments: list[dict[str, Any]],
    ) -> None:
        """Apply assertion adjustments to contract.

        Args:
            contract: Contract to modify.
            adjustments: List of adjustment specifications.
        """
        assertions = contract.get("assertions", [])
        adjustment_map: dict[str, dict[str, Any]] = {}

        for adj in adjustments:
            for assertion_id in adj.get("affected_assertions", []):
                adjustment_map[assertion_id] = adj

        for assertion in assertions:
            assertion_id = assertion.get("id")
            if assertion_id in adjustment_map:
                adj = adjustment_map[assertion_id]
                self._apply_single_adjustment(assertion, adj)

        contract["assertions"] = assertions

    def _apply_single_adjustment(
        self,
        assertion: dict[str, Any],
        adjustment: dict[str, Any],
    ) -> None:
        """Apply a single adjustment to an assertion.

        Args:
            assertion: Assertion to modify.
            adjustment: Adjustment specification.
        """
        adj_type = adjustment.get("type")
        params = assertion.setdefault("parameters", {})

        if adj_type == "color_tolerance":
            current = params.get("delta_e", 10)
            params["delta_e"] = current + adjustment.get("delta_e_increase", 5)

        elif adj_type == "temporal_tolerance":
            current = params.get("frame_window", 1)
            params["frame_window"] = current + adjustment.get("frame_window_increase", 2)

        elif adj_type == "temporal_window":
            current = params.get("window_size", 1)
            multiplier = adjustment.get("window_size_multiplier", 1.5)
            params["window_size"] = int(current * multiplier)

        elif adj_type == "occlusion_handling":
            config = adjustment.get("config", {})
            params["allow_partial_occlusion"] = config.get("allow_partial_occlusion", True)
            params["min_visibility_threshold"] = config.get("min_visibility_threshold", 0.3)

        elif adj_type == "temporal_smoothing":
            params["smoothing_window"] = adjustment.get("window_size", 5)

        elif adj_type == "artifact_filter":
            params["artifact_filter"] = adjustment.get("filter_config", {})

    async def _reevaluate(
        self,
        contract: dict[str, Any],
        context: dict[str, Any],
    ) -> list[Failure]:
        """Re-run evaluation to check if fixes worked.

        This is a placeholder that would integrate with the actual
        evaluation system. In practice, this would trigger a re-run
        of the evaluation pipeline.

        Args:
            contract: The updated contract.
            context: Context for evaluation.

        Returns:
            List of remaining failures after re-evaluation.
        """
        return []

    async def diagnose_only(
        self,
        evaluation_result: EvaluationResult,
        context: dict[str, Any] | None = None,
    ) -> list[DiagnosisResult]:
        """Diagnose failures without generating fixes.

        Args:
            evaluation_result: The evaluation result containing failures.
            context: Optional additional context for diagnosis.

        Returns:
            List of diagnosis results.
        """
        if context is None:
            context = {}

        failures = evaluation_result.failures
        return await self._diagnose_failures(failures, context)

    async def generate_fixes_only(
        self,
        diagnoses: list[DiagnosisResult],
        contract: dict[str, Any],
    ) -> list[FixProposal]:
        """Generate fix proposals without applying them.

        Args:
            diagnoses: Diagnoses to generate fixes for.
            contract: The current contract.

        Returns:
            List of fix proposals.
        """
        return await self._generate_fixes(diagnoses, contract)


def create_pipeline(
    config: PipelineConfig | None = None,
) -> SelfHealingPipeline:
    """Create a configured self-healing pipeline.

    Args:
        config: Optional pipeline configuration.

    Returns:
        A configured SelfHealingPipeline instance.
    """
    return SelfHealingPipeline(config=config)
