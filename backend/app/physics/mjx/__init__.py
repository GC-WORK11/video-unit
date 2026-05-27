"""
MJX: MuJoCo in JAX (DeepMind)
==============================

Wrapper for DeepMind's MJX - MuJoCo physics engine in JAX.
"""

from .backprop_sim import BackpropMJX

__all__ = [
    "BackpropMJX",
]
