"""Result types for VideoUnit evaluators."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvidenceFrame:
    """A single frame of evidence supporting an evaluation failure or pass.

    Attributes:
        timestamp: ISO format timestamp string (e.g., "00:01:23.456").
        frame_number: Zero-indexed frame number in the video.
        thumbnail_path: Path to the extracted frame thumbnail image.
        bbox: Optional bounding box [x1, y1, x2, y2] for highlighted regions.
        explanation: Human-readable explanation of what this frame shows.
        confidence: Model confidence in the analysis (0.0 to 1.0).
    """

    timestamp: str
    frame_number: int
    thumbnail_path: str
    bbox: Optional[list[float]] = None
    explanation: str = ""
    confidence: float = 1.0


@dataclass
class Failure:
    """Represents a single failure detected during evaluation.

    Attributes:
        timestamp: ISO format timestamp string when the failure occurred.
        frame_number: Zero-indexed frame number when the failure was detected.
        type: Failure type identifier (e.g., "object_missing", "color_shift").
        severity: Severity level - one of "info", "warning", "fail", "critical".
        message: Human-readable failure description.
        object: Optional object name/id if failure relates to a specific object.
        suggested_fix: Optional suggested remediation for this failure.
    """

    timestamp: str
    frame_number: int
    type: str
    severity: str
    message: str
    object: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of running an evaluator on a video.

    Attributes:
        passed: True if the video passed all checks for this evaluator.
        score: Overall quality score from 0-100 (100 = perfect).
        failures: List of detected failures, empty if passed.
        evidence: List of evidence frames supporting the result.
    """

    passed: bool
    score: float
    failures: list[Failure] = field(default_factory=list)
    evidence: list[EvidenceFrame] = field(default_factory=list)

    def add_failure(
        self,
        timestamp: str,
        frame_number: int,
        failure_type: str,
        severity: str,
        message: str,
        object: Optional[str] = None,
        suggested_fix: Optional[str] = None,
    ) -> None:
        """Add a failure to this result.

        Args:
            timestamp: When the failure occurred.
            frame_number: Frame where failure was detected.
            failure_type: Type identifier for the failure.
            severity: One of "info", "warning", "fail", "critical".
            message: Description of the failure.
            object: Optional related object name.
            suggested_fix: Optional remediation suggestion.
        """
        self.failures.append(
            Failure(
                timestamp=timestamp,
                frame_number=frame_number,
                type=failure_type,
                severity=severity,
                message=message,
                object=object,
                suggested_fix=suggested_fix,
            )
        )
        self.passed = False

    def add_evidence(
        self,
        timestamp: str,
        frame_number: int,
        thumbnail_path: str,
        explanation: str,
        bbox: Optional[list[float]] = None,
        confidence: float = 1.0,
    ) -> None:
        """Add an evidence frame to this result.

        Args:
            timestamp: Frame timestamp.
            frame_number: Frame number.
            thumbnail_path: Path to thumbnail image.
            explanation: What this frame demonstrates.
            bbox: Optional bounding box to highlight.
            confidence: Confidence in the analysis.
        """
        self.evidence.append(
            EvidenceFrame(
                timestamp=timestamp,
                frame_number=frame_number,
                thumbnail_path=thumbnail_path,
                bbox=bbox,
                explanation=explanation,
                confidence=confidence,
            )
        )
