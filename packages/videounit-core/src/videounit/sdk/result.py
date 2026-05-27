"""Result types and utilities for VideoUnit evaluation."""

from typing import Optional

from videounit.sdk.models import EvaluationResult, Failure, EvidenceFrame, Severity


class ResultFormatter:
    """Format evaluation results for display."""

    @staticmethod
    def summary(result: EvaluationResult) -> str:
        """Generate a human-readable summary of the evaluation result.

        Args:
            result: The evaluation result.

        Returns:
            Formatted summary string.
        """
        lines = [
            f"Evaluation Result: {result.overall:.1%}",
            f"Confidence: {result.confidence:.1%}",
            f"Failures: {result.num_failures} ({result.critical_failures} critical)",
            "",
        ]

        if result.failures:
            lines.append("Failures:")
            for failure in result.failures:
                severity_icon = {
                    Severity.INFO: "i",
                    Severity.WARNING: "!",
                    Severity.FAIL: "X",
                    Severity.CRITICAL: "!!",
                }.get(failure.severity, "?")
                lines.append(f"  [{severity_icon}] {failure.message}")
                if failure.object:
                    lines.append(f"      Object: {failure.object}")
        else:
            lines.append("All assertions passed.")

        return "\n".join(lines)

    @staticmethod
    def get_failures_by_severity(
        result: EvaluationResult, severity: Severity
    ) -> list[Failure]:
        """Get failures filtered by severity.

        Args:
            result: The evaluation result.
            severity: The severity level to filter by.

        Returns:
            List of matching failures.
        """
        return [f for f in result.failures if f.severity == severity]

    @staticmethod
    def get_critical_failures(result: EvaluationResult) -> list[Failure]:
        """Get all critical severity failures.

        Args:
            result: The evaluation result.

        Returns:
            List of critical failures.
        """
        return ResultFormatter.get_failures_by_severity(result, Severity.CRITICAL)

    @staticmethod
    def to_json_dict(result: EvaluationResult) -> dict:
        """Convert result to a JSON-serializable dictionary.

        Args:
            result: The evaluation result.

        Returns:
            Dictionary representation.
        """
        return {
            "overall": result.overall,
            "categories": result.categories,
            "confidence": result.confidence,
            "num_failures": result.num_failures,
            "critical_failures": result.critical_failures,
            "passed": result.passed,
            "failures": [f.model_dump(mode="json") for f in result.failures],
            "evidence_count": len(result.evidence),
            "run_id": result.run_id,
        }
