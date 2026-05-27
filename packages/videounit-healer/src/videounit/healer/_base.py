"""VideoUnit Self-Healing Pipeline - Base Classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from videounit_evaluators._result import Failure


class RepairStrategy(Enum):
    """Strategies for repairing test failures."""

    PROMPT_ENHANCEMENT = "prompt_enhancement"
    TOLERANCE_ADJUSTMENT = "tolerance_adjustment"
    ASSERTION_ADDITION = "assertion_addition"
    FULL_REGENERATION = "full_regeneration"


@dataclass
class DiagnosisResult:
    """Result of diagnosing a test failure.

    Attributes:
        root_cause: The identified root cause of the failure.
        evidence: Evidence supporting the diagnosis (e.g., frame analysis, metrics).
        confidence: Confidence score from 0.0 to 1.0 in the diagnosis.
        related_failures: List of other failure IDs that may share this root cause.
        repair_strategy: Recommended strategy for fixing this failure.
    """

    root_cause: str
    evidence: dict[str, Any]
    confidence: float
    related_failures: list[str] = field(default_factory=list)
    repair_strategy: RepairStrategy = RepairStrategy.PROMPT_ENHANCEMENT


@dataclass
class FixProposal:
    """A proposed fix for a test failure.

    Attributes:
        proposal_id: Unique identifier for this proposal.
        target_failure_type: The type of failure this proposal addresses.
        prompt_delta: The delta/patch to apply to the original prompt.
        contract_modifications: Changes to make to the contract assertions.
        confidence: Confidence score from 0.0 to 1.0 in this fix.
        risks: List of potential risks or side effects of this fix.
    """

    proposal_id: str
    target_failure_type: str
    prompt_delta: str
    contract_modifications: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    risks: list[str] = field(default_factory=list)


@dataclass
class SelfHealingReport:
    """Report summarizing the self-healing process for a test run.

    Attributes:
        total_failures: Total number of failures detected.
        diagnoses: List of diagnosis results, one per failure type.
        fix_proposals: List of proposed fixes for each diagnosis.
        applied_fixes: List of fixes that were successfully applied.
        failed_fixes: List of fixes that could not be applied.
        iterations: Number of healing iterations performed.
        converged: Whether the healing process converged (no more failures).
    """

    total_failures: int
    diagnoses: list[DiagnosisResult] = field(default_factory=list)
    fix_proposals: list[FixProposal] = field(default_factory=list)
    applied_fixes: list[FixProposal] = field(default_factory=list)
    failed_fixes: list[tuple[FixProposal, str]] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False


class FailureAnalyzer(ABC):
    """Abstract base class for failure analyzers.

    Analyzers are pluggable components that handle specific types of failures.
    Subclass this to create analyzers for different failure categories.
    """

    failure_type: str

    @abstractmethod
    async def analyze(
        self, failure: Failure, context: dict[str, Any]
    ) -> DiagnosisResult:
        """Analyze a failure and return a diagnosis.

        Args:
            failure: The failure to analyze.
            context: Additional context for the analysis (e.g., video metadata,
                previous evaluation results, perception outputs).

        Returns:
            A DiagnosisResult with the root cause and evidence.
        """
        pass

    @abstractmethod
    def can_handle(self, failure: Failure) -> bool:
        """Check if this analyzer can handle the given failure.

        Args:
            failure: The failure to check.

        Returns:
            True if this analyzer can handle this failure type.
        """
        pass


class FixGenerator(ABC):
    """Abstract base class for fix proposal generators.

    Generators create fix proposals based on diagnoses. Subclass this to
    create generators for different repair strategies.
    """

    strategy: RepairStrategy

    @abstractmethod
    async def generate(
        self, diagnosis: DiagnosisResult, contract: dict[str, Any]
    ) -> FixProposal:
        """Generate a fix proposal based on a diagnosis.

        Args:
            diagnosis: The diagnosis result to generate a fix for.
            contract: The current contract being tested.

        Returns:
            A FixProposal with the suggested fix.
        """
        pass


class DiagnosisEngine:
    """Engine that coordinates failure analysis using pluggable analyzers.

    The engine maintains a registry of analyzers and routes failures to
    the appropriate analyzer based on failure type.

    Attributes:
        analyzers: List of registered FailureAnalyzer instances.
    """

    def __init__(self) -> None:
        self.analyzers: list[FailureAnalyzer] = []

    def register(self, analyzer: FailureAnalyzer) -> None:
        """Register an analyzer with the engine.

        Args:
            analyzer: The analyzer to register.
        """
        self.analyzers.append(analyzer)

    async def diagnose(
        self, failure: Failure, context: dict[str, Any]
    ) -> DiagnosisResult:
        """Diagnose a failure using the appropriate analyzer.

        Args:
            failure: The failure to diagnose.
            context: Additional context for diagnosis.

        Returns:
            A DiagnosisResult with the root cause.

        Raises:
            ValueError: If no analyzer can handle the failure type.
        """
        for analyzer in self.analyzers:
            if analyzer.can_handle(failure):
                return await analyzer.analyze(failure, context)

        return DiagnosisResult(
            root_cause="unknown",
            evidence={"original_failure_type": failure.type},
            confidence=0.0,
            repair_strategy=RepairStrategy.FULL_REGENERATION,
        )

    async def diagnose_batch(
        self,
        failures: list[Failure],
        context: dict[str, Any],
    ) -> list[DiagnosisResult]:
        """Diagnose multiple failures.

        Args:
            failures: List of failures to diagnose.
            context: Additional context for diagnosis.

        Returns:
            List of DiagnosisResults, one per failure.
        """
        results = []
        for failure in failures:
            result = await self.diagnose(failure, context)
            results.append(result)
        return results


class FixProposalGenerator:
    """Generator that creates fix proposals based on diagnoses.

    Coordinates multiple fix generators and selects the appropriate one
    based on the diagnosis repair strategy.

    Attributes:
        generators: List of registered FixGenerator instances.
    """

    def __init__(self) -> None:
        self.generators: list[FixGenerator] = []

    def register(self, generator: FixGenerator) -> None:
        """Register a fix generator.

        Args:
            generator: The generator to register.
        """
        self.generators.append(generator)

    async def generate(
        self, diagnosis: DiagnosisResult, contract: dict[str, Any]
    ) -> FixProposal:
        """Generate a fix proposal for a diagnosis.

        Args:
            diagnosis: The diagnosis to generate a fix for.
            contract: The current contract.

        Returns:
            A FixProposal with the suggested fix.

        Raises:
            ValueError: If no generator can handle the repair strategy.
        """
        for generator in self.generators:
            if generator.strategy == diagnosis.repair_strategy:
                return await generator.generate(diagnosis, contract)

        return FixProposal(
            proposal_id=f"fallback-{diagnosis.root_cause}",
            target_failure_type=diagnosis.root_cause,
            prompt_delta="",
            confidence=0.0,
            risks=["No specific generator found for strategy"],
        )

    async def generate_batch(
        self,
        diagnoses: list[DiagnosisResult],
        contract: dict[str, Any],
    ) -> list[FixProposal]:
        """Generate fix proposals for multiple diagnoses.

        Args:
            diagnoses: List of diagnoses to generate fixes for.
            contract: The current contract.

        Returns:
            List of FixProposals, one per diagnosis.
        """
        proposals = []
        for diagnosis in diagnoses:
            proposal = await self.generate(diagnosis, contract)
            proposals.append(proposal)
        return proposals
