"""
AETHER Scene Graph Module
========================

Kinematic discovery and scene graph construction.

Submodules:
- splart: V-NEXT 3D Gaussian Splatting for articulated objects
- real2code: LLM-guided URDF generation from kinematic analysis
- legacy_compat: Compatibility layer for switching implementations
"""

# V-NEXT modules
from .splart import (
    GaussianCloud,
    GaussianCloudReconstructor,
    SPLARTReconstructor,
    discover_kinematic_structure,
)

from .real2code import (
    JointDetector,
    JointType,
    JointDetectionResult,
    URDFCompiler,
    LinkSpec,
    JointSpec,
)

from .legacy_compat import (
    discover_kinematic_structure,
    kinematic_tree_to_mjcf,
    set_implementation,
    get_implementation,
    VNEXT_AVAILABLE,
    LEGACY_AVAILABLE,
)

# Legacy (kept for compatibility)
try:
    from .kinematic_discovery import (
        discover_kinematic_structure as legacy_discover,
        kinematic_tree_to_mjcf as legacy_to_mjcf,
    )
except ImportError:
    legacy_discover = None
    legacy_to_mjcf = None

# Schema (only import what's available)
try:
    from .schema import RigidBody, Joint, KinematicChain
    SceneGraph = None  # Not defined yet
except ImportError:
    RigidBody = None
    Joint = None
    KinematicChain = None
    SceneGraph = None

__all__ = [
    # V-NEXT
    "GaussianCloud",
    "GaussianCloudReconstructor",
    "SPLARTReconstructor",
    "JointDetector",
    "JointType",
    "JointDetectionResult",
    "URDFCompiler",
    "LinkSpec",
    "JointSpec",
    
    # Main API
    "discover_kinematic_structure",
    "kinematic_tree_to_mjcf",
    "set_implementation",
    "get_implementation",
    "VNEXT_AVAILABLE",
    "LEGACY_AVAILABLE",
    
    # Legacy
    "legacy_discover",
    "legacy_to_mjcf",
    
    # Schema
    "SceneGraph",
    "RigidBody", 
    "Joint",
    "KinematicChain",
]
