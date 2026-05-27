"""Base classes for VideoUnit evaluators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._context import EvaluationContext
    from ._result import EvaluationResult


class Evaluator(ABC):
    """Base class for all VideoUnit evaluators.

    Each evaluator checks one type of assertion against a video contract
    and returns failures and a score indicating how well the video passes.

    Attributes:
        name: Unique identifier for this evaluator type.
        required_inputs: List of input types needed (e.g. ["tracks", "masks", "frames"]).
        graceful_on_missing: If True, missing inputs cause score=0 with warning rather than exception.
    """

    name: str
    required_inputs: list[str] = []
    graceful_on_missing: bool = True

    @abstractmethod
    async def run(self, context: "EvaluationContext") -> "EvaluationResult":
        """Run the evaluator on the given context.

        Args:
            context: The evaluation context containing video data and perception results.

        Returns:
            EvaluationResult with passed/fail status, score, failures, and evidence.
        """
        pass

    def get_missing_inputs(self, context: "EvaluationContext") -> set[str]:
        """Check which required inputs are missing from context.

        Args:
            context: The evaluation context to validate.

        Returns:
            Set of missing input names.
        """
        return set(self.required_inputs) - set(context.available_inputs)

    def validate_inputs(self, context: "EvaluationContext") -> None:
        """Validate that required inputs are present in context.

        Args:
            context: The evaluation context to validate.

        Raises:
            ValueError: If a required input is missing and graceful_on_missing is False.
        """
        missing = self.get_missing_inputs(context)
        if missing and not self.graceful_on_missing:
            raise ValueError(
                f"Evaluator '{self.name}' requires inputs {list(missing)} "
                f"but only {context.available_inputs} are available"
            )
