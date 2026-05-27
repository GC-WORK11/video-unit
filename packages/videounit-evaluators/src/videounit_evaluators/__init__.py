"""VideoUnit Evaluators - Assertion plugins for video testing."""

from videounit_evaluators._base import Evaluator
from videounit_evaluators._context import EvaluationContext
from videounit_evaluators._result import EvaluationResult, Failure, EvidenceFrame
from videounit_evaluators._registry import get_evaluator, all_evaluators, register_evaluator

# Import all evaluators to trigger @register_evaluator decorators
from videounit_evaluators import (
    color_constant,
    motion_direction,
    object_exists,
    object_persistence,
    physics_sanity,
    scene_cut,
    temporal_flicker,
    vlm_ensemble,
    vlm_judge,
)

__all__ = [
    "Evaluator",
    "EvaluationContext",
    "EvaluationResult",
    "Failure",
    "EvidenceFrame",
    "get_evaluator",
    "all_evaluators",
    "register_evaluator",
]
