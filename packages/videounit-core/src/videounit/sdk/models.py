"""VideoUnit Pydantic models for contracts and evaluation results."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    FAIL = "fail"
    CRITICAL = "critical"


class AssertionType(str, Enum):
    OBJECT_EXISTS = "object_exists"
    OBJECT_COLOR_CONSTANT = "object_color_constant"
    MOTION_DIRECTION = "motion_direction"
    MOTION_SPEED = "motion_speed"
    OBJECT_PERSISTENCE = "object_persistence"
    SCENE_TRANSITION = "scene_transition"
    COUNT_CONSTANT = "count_constant"
    TRAJECTORY = "trajectory"


class PhaseSpec(BaseModel):
    """Specification for a temporal phase in the video."""

    start: float = Field(description="Start time in seconds")
    end: float = Field(description="End time in seconds")
    name: Optional[str] = Field(None, description="Phase name")


class PersistenceSpec(BaseModel):
    """Specification for object persistence tracking."""

    min_frames: int = Field(default=1, description="Minimum frames object must persist")
    max_gap: float = Field(default=0.0, description="Maximum gap between detections")


class TestMetadata(BaseModel):
    """Metadata about the test."""

    name: str
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = Field(default_factory=list)


class InputSpec(BaseModel):
    """Specification for video input."""

    source: str = Field(description="Video file path or URL")
    fps: Optional[float] = Field(None, description="Expected FPS (None = detect)")
    duration: Optional[float] = Field(None, description="Expected duration in seconds")


class ObjectSpec(BaseModel):
    """Specification for an object to track."""

    id: str
    name: str
    attributes: dict = Field(default_factory=dict)
    persistence: Optional[PersistenceSpec] = None
    expected_color: Optional[str] = None
    expected_bbox: Optional[list[float]] = None


class Assertion(BaseModel):
    """An assertion to validate in the video."""

    type: AssertionType
    object: Optional[str] = Field(None, description="Object ID to check")
    from_: Optional[str] = Field(None, alias="from", description="Source object ID")
    to: Optional[str] = Field(None, description="Target object ID")
    expected_color: Optional[str] = None
    phase: Optional[PhaseSpec] = None
    tolerance: Optional[dict] = None
    expected_value: Optional[str] = None

    model_config = {"populate_by_name": True}


class ScoringSpec(BaseModel):
    """Scoring configuration for evaluation."""

    weights: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    categories: list[str] = Field(default_factory=list)


class VideoContract(BaseModel):
    """VideoUnit contract specifying test requirements."""

    version: str = "0.1"
    test: TestMetadata
    input: InputSpec
    objects: list[ObjectSpec] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)
    scoring: Optional[ScoringSpec] = None

    @classmethod
    def from_yaml(cls, path: str) -> "VideoContract":
        """Load contract from YAML file."""
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoContract":
        """Create contract from dictionary."""
        return cls(**data)


class Failure(BaseModel):
    """A failed assertion with evidence."""

    timestamp: str
    frame: int
    type: str
    severity: Severity
    message: str
    object: Optional[str] = None
    suggested_fix: Optional[str] = None


class EvidenceFrame(BaseModel):
    """Frame evidence for a result."""

    timestamp: str
    frame_number: int
    thumbnail_path: str
    bbox: Optional[list[float]] = None
    mask_overlay_path: Optional[str] = None
    explanation: str
    confidence: float


class ObjectTrack(BaseModel):
    """Track of an object across frames."""

    object_id: str
    frames: list[int] = Field(default_factory=list)
    timestamps: list[float] = Field(default_factory=list)
    centroids: list[list[float]] = Field(default_factory=list)
    bboxes: list[list[float]] = Field(default_factory=list)
    confidence: float = 1.0


class Score(BaseModel):
    """Individual score component."""

    category: str
    value: float
    weight: float = 1.0
    passed: bool = True


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    overall: float = Field(description="Overall score 0.0 - 1.0")
    categories: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(description="Confidence in the evaluation")
    num_failures: int = 0
    critical_failures: int = 0
    failures: list[Failure] = Field(default_factory=list)
    evidence: list[EvidenceFrame] = Field(default_factory=list)
    tracks: list[ObjectTrack] = Field(default_factory=list)
    run_id: Optional[str] = None
    video_path: Optional[str] = None
    duration_s: Optional[float] = None

    @property
    def passed(self) -> bool:
        """Return True if evaluation passed (no failures)."""
        return self.num_failures == 0
