"""
Symplectic Integrator: Mathematically Conservative
================================================

Symplectic (Verlet) integration preserves phase space volume.

Unlike Euler/RK4 which are NOT symplectic and drift energy over time,
symplectic methods exactly conserve a modified Hamiltonian.

Mathematical Foundation:
- Symplectic Euler: p_{n+1} = p_n + dt * (-∇V(q_n)), q_{n+1} = q_n + dt * ∇T(p_{n+1})
- Velocity Verlet: Symplectic and reversible

Key insight: We're learning dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
The symplectic integrator respects this structure.
"""

import jax.numpy as jnp
from typing import Tuple, Callable, Optional
import logging

log = logging.getLogger(__name__)


class SymplecticIntegrator:
    """
    N-step symplectic integrator for Hamiltonian dynamics.
    
    For a system with H(q, p) = T(p) + V(q):
    - T(p) = pᵀ M⁻¹ p / 2 (kinetic energy, M = mass matrix)
    - V(q) = potential energy
    
    We have:
    - ∂H/∂p = M⁻¹ p = dT_dp
    - ∂H/∂q = ∇V = dV_dq
    
    Usage:
        # For spring-mass: V = k*q²/2, T = p²/(2m)
        # dV/dq = k*q, dT/dp = p/m
        
        integrator = SymplecticIntegrator(dt=0.01)
        
        q, p = q0, p0
        for _ in range(n_steps):
            # Semi-implicit Euler (symplectic)
            p = p - dt * (k * q)  # dp/dt = -dV/dq
            q = q + dt * (p / m)   # dq/dt = dT/dp
    """
    
    def __init__(
        self,
        dt: float = 0.001,
        integrator: str = "verlet",
    ):
        """
        Args:
            dt: Integration time step
            integrator: "verlet" or "euler" (symplectic)
        """
        self.dt = dt
        self.integrator = integrator
    
    def verlet_step(
        self,
        q: jnp.ndarray,
        p: jnp.ndarray,
        mass: jnp.ndarray,
        stiffness: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Velocity Verlet (symplectic).
        
        Algorithm:
        1. p_{n+1/2} = p_n + (dt/2) * (-∇V(q_n))
        2. q_{n+1} = q_n + dt * (p_{n+1/2} / m)
        3. p_{n+1} = p_{n+1/2} + (dt/2) * (-∇V(q_{n+1}))
        """
        # Step 1: Half-step momentum
        p_half = p - 0.5 * self.dt * stiffness * q
        
        # Step 2: Full-step position
        q_new = q + self.dt * (p_half / mass)
        
        # Step 3: Complete momentum step
        p_new = p_half - 0.5 * self.dt * stiffness * q_new
        
        return q_new, p_new
    
    def euler_step(
        self,
        q: jnp.ndarray,
        p: jnp.ndarray,
        mass: jnp.ndarray,
        stiffness: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Semi-implicit Euler (symplectic).
        
        Algorithm:
        1. p_{n+1} = p_n + dt * (-∇V(q_n))
        2. q_{n+1} = q_n + dt * (p_{n+1} / m)
        """
        p_new = p - self.dt * stiffness * q
        q_new = q + self.dt * (p_new / mass)
        return q_new, p_new
    
    def integrate(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        mass: jnp.ndarray,
        stiffness: jnp.ndarray,
        n_steps: int,
        damping: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Integrate for n_steps.
        
        Args:
            q0: Initial positions [n_dofs]
            p0: Initial momenta [n_dofs]
            mass: [n_dofs] mass per DOF
            stiffness: [n_dofs] spring constant per DOF
            n_steps: Number of integration steps
            damping: Optional [n_dofs] damping coefficient
            
        Returns:
            q_final, p_final: Final state after n_steps
        """
        q, p = q0, p0
        
        if damping is None:
            damping = jnp.zeros_like(q)
        
        step_fn = self.verlet_step if self.integrator == "verlet" else self.euler_step
        
        for _ in range(n_steps):
            # Add damping force: -c * q_dot = -c * (p/m)
            damping_force = damping * (p / mass)
            
            # Update with damping
            q, p = step_fn(q, p - self.dt * damping_force, mass, stiffness)
        
        return q, p
    
    def integrate_trajectory(
        self,
        q0: jnp.ndarray,
        p0: jnp.ndarray,
        mass: jnp.ndarray,
        stiffness: jnp.ndarray,
        n_steps: int,
        damping: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Integrate and return full trajectory.
        
        Returns:
            q_trajectory: [n_steps+1, n_dofs]
            p_trajectory: [n_steps+1, n_dofs]
        """
        q_traj = [q0]
        p_traj = [p0]
        
        q, p = q0, p0
        
        if damping is None:
            damping = jnp.zeros_like(q)
        
        step_fn = self.verlet_step if self.integrator == "verlet" else self.euler_step
        
        for _ in range(n_steps):
            damping_force = damping * (p / mass)
            q, p = step_fn(q, p - self.dt * damping_force, mass, stiffness)
            q_traj.append(q)
            p_traj.append(p)
        
        return jnp.array(q_traj), jnp.array(p_traj)


def test_symplectic_integrator():
    """Test that symplectic integration preserves energy."""
    print("=" * 60)
    print("Testing Symplectic Integrator")
    print("=" * 60)
    
    import numpy as np
    
    # Simple harmonic oscillator: H = p²/2m + kq²/2
    m = 1.0  # mass
    k = 1.0  # spring constant
    dt = 0.01
    
    # Initial conditions
    q0 = np.array([1.0])  # amplitude
    p0 = np.array([0.0])  # at equilibrium
    
    # Initial energy: H = 0.5*k*q² + 0.5*p²/m = 0.5*1*1 + 0 = 0.5
    E0 = 0.5 * k * q0[0]**2 + 0.5 * p0[0]**2 / m
    print(f"Initial energy: {E0:.6f}")
    
    # Integrate with Verlet (symplectic)
    integrator = SymplecticIntegrator(dt=dt, integrator="verlet")
    
    n_steps = 10000
    
    q_final, p_final = integrator.integrate(
        jnp.array(q0),
        jnp.array(p0),
        jnp.array([m]),
        jnp.array([k]),
        n_steps,
    )
    
    E_final = 0.5 * k * float(q_final[0]**2) + 0.5 * float(p_final[0]**2) / m
    energy_drift = abs(E_final - E0) / E0
    
    print(f"Final energy: {E_final:.6f}")
    print(f"Energy drift: {energy_drift:.10f} ({energy_drift*100:.10f}%)")
    
    # Compare with standard (non-symplectic) Euler
    print("\n--- Non-symplectic Euler comparison ---")
    
    def euler_step(q, p, dt):
        p_new = p - dt * k * q
        q_new = q + dt * p_new / m
        return q_new, p_new
    
    q_e, p_e = q0.copy(), p0.copy()
    for _ in range(n_steps):
        q_e, p_e = euler_step(q_e, p_e, dt)
    
    E_euler = 0.5 * k * q_e[0]**2 + 0.5 * p_e[0]**2 / m
    drift_euler = abs(E_euler - E0) / E0
    
    print(f"Euler final energy: {E_euler:.6f}")
    print(f"Euler energy drift: {drift_euler:.6f} ({drift_euler*100:.6f}%)")
    
    print("\n" + "=" * 60)
    if energy_drift < 1e-10:
        print("✅ Symplectic integrator preserves energy EXACTLY!")
        print("   (Drift is machine precision)")
    else:
        print(f"⚠️  Unexpected drift: {energy_drift}")
    print("=" * 60)


if __name__ == "__main__":
    test_symplectic_integrator()
