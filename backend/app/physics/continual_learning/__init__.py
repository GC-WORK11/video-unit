"""
AETHER Continual Learning: K-FAC EWC
====================================

Continual learning implementation using Kronecker-Factored Approximate Curvature.
"""

from .kfac_fisher import KFACEstimator, KFACLayerFactors, KFACEWCRegularizer

__all__ = [
    "KFACEstimator",
    "KFACLayerFactors",
    "KFACEWCRegularizer",
]
