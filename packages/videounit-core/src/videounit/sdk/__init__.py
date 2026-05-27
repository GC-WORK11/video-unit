"""SDK module for VideoUnit client and models."""

from videounit.sdk.client import VideoUnitClient
from videounit.sdk.models import (
    VideoContract,
    TestMetadata,
    InputSpec,
    ObjectSpec,
    Assertion,
    PhaseSpec,
    PersistenceSpec,
    ScoringSpec,
    EvaluationResult,
    Failure,
    EvidenceFrame,
    ObjectTrack,
    Score,
)

__all__ = [
    "VideoUnitClient",
    "VideoContract",
    "TestMetadata",
    "InputSpec",
    "ObjectSpec",
    "Assertion",
    "PhaseSpec",
    "PersistenceSpec",
    "ScoringSpec",
    "EvaluationResult",
    "Failure",
    "EvidenceFrame",
    "ObjectTrack",
    "Score",
]
