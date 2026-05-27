"""
Legacy Compatibility Layer
========================

Allows seamless switching between:
- V-NEXT SPLART implementation (new)
- Old spectral clustering implementation (legacy)

Usage:
    from app.scene_graph.legacy_compat import discover_kinematic_structure
    
    # Automatically uses V-NEXT
    kin_tree = discover_kinematic_structure(tracks_3d, n_bodies=2)

To force legacy:
    from app.scene_graph.legacy_compat import discover_kinematic_structure_legacy
"""

import importlib
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Try to import V-NEXT implementation first
VNEXT_AVAILABLE = False
try:
    from .splart.reconstruct import (
        discover_kinematic_structure as vnext_discover,
        kinematic_tree_to_mjcf as vnext_to_mjcf,
        SPLARTReconstructor,
        KinematicTree,
        LegacyKinematicTree,
    )
    VNEXT_AVAILABLE = True
    log.info("V-NEXT SPLART implementation available")
except ImportError as e:
    log.warning(f"V-NEXT SPLART import failed: {e}")
    vnext_discover = None
    vnext_to_mjcf = None
    SPLARTReconstructor = None
    KinematicTree = None
    LegacyKinematicTree = None

# Try to import legacy implementation
LEGACY_AVAILABLE = False
try:
    from .kinematic_discovery import (
        discover_kinematic_structure as legacy_discover,
        kinematic_tree_to_mjcf as legacy_to_mjcf,
    )
    LEGACY_AVAILABLE = True
    log.info("Legacy spectral clustering implementation available")
except ImportError as e:
    log.warning(f"Legacy kinematic_discovery import failed: {e}")
    legacy_discover = None
    legacy_to_mjcf = None


# Global flag for switching implementations
_USE_VNEXT = VNEXT_AVAILABLE  # Default to V-NEXT if available


def set_implementation(which: str = "vnext"):
    """
    Set which implementation to use.
    
    Args:
        which: "vnext" or "legacy"
    """
    global _USE_VNEXT
    
    if which == "vnext" and not VNEXT_AVAILABLE:
        log.error("V-NEXT not available")
        return False
    
    if which == "legacy" and not LEGACY_AVAILABLE:
        log.error("Legacy not available")
        return False
    
    _USE_VNEXT = (which == "vnext")
    log.info(f"Using {which} implementation")
    return True


def get_implementation() -> str:
    """Get current implementation name."""
    return "vnext" if _USE_VNEXT else "legacy"


# Main API - automatically uses configured implementation
def discover_kinematic_structure(tracks_3d, n_bodies: int = 2):
    """
    Unified entry point for kinematic discovery.
    
    Automatically uses V-NEXT if available, falls back to legacy.
    
    Args:
        tracks_3d: [T, N, 3] 3D trajectories
        n_bodies: Estimated number of rigid bodies
        
    Returns:
        KinematicTree-compatible object
    """
    if _USE_VNEXT and vnext_discover:
        return vnext_discover(tracks_3d, n_bodies)
    elif legacy_discover:
        return legacy_discover(tracks_3d, n_bodies)
    else:
        raise RuntimeError("No kinematic discovery implementation available!")


def kinematic_tree_to_mjcf(kin_tree):
    """
    Convert kinematic tree to MuJoCo MJCF XML.
    
    Args:
        kin_tree: KinematicTree from discover_kinematic_structure
        
    Returns:
        MJCF XML string
    """
    if _USE_VNEXT and vnext_to_mjcf:
        return vnext_to_mjcf(kin_tree)
    elif legacy_to_mjcf:
        return legacy_to_mjcf(kin_tree)
    else:
        raise RuntimeError("No kinematic discovery implementation available!")


# Explicit legacy wrapper
def discover_kinematic_structure_legacy(tracks_3d, n_bodies: int = 2):
    """
    Force use of legacy spectral clustering implementation.
    """
    if not LEGACY_AVAILABLE:
        raise RuntimeError("Legacy implementation not available!")
    return legacy_discover(tracks_3d, n_bodies)


# Export V-NEXT specific classes if available
__all__ = [
    "discover_kinematic_structure",
    "kinematic_tree_to_mjcf",
    "discover_kinematic_structure_legacy",
    "set_implementation",
    "get_implementation",
    "VNEXT_AVAILABLE",
    "LEGACY_AVAILABLE",
    "SPLARTReconstructor",
    "KinematicTree",
    "LegacyKinematicTree",
]
