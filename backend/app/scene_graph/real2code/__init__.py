"""
Real2Code: LLM-guided URDF Generation
=====================================

V-NEXT replacement for aspect-ratio-based mechanism detection.

Uses:
- Pure mathematical joint classification (DOF analysis)
- Structured URDF compilation from kinematic analysis
- LLM-guided parameter suggestions
"""

from .joint_detector import JointDetector, JointType, JointDetectionResult
from .urdf_compiler import URDFCompiler, LinkSpec, JointSpec

__all__ = [
    "JointDetector",
    "JointType", 
    "JointDetectionResult",
    "URDFCompiler",
    "LinkSpec",
    "JointSpec",
]
