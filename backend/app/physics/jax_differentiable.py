"""
AETHER JAX Differentiable Physics Engine (Phase 2.1)
===================================================

Implements differentiable physics simulation using JAX for automatic differentiation.
This replaces the simple Euler loop with a proper differentiable simulator.

THE BREAKTHROUGH:
Instead of just fitting parameters to trajectories, we now:
1. Simulate physics FORWARD in JAX
2. Compare with observed trajectory
3. Use jax.grad to optimize parameters BACKWARD through the simulation

This means the physics engine itself is differentiable - we can learn
friction, restitution, and mass by backpropagating through the simulation.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging

log = logging.getLogger(__name__)


class DifferentiableSimulator:
    """
    JAX-based differentiable physics simulator.
    
    Uses symplectic integration (Verlet) for energy conservation,
    with automatic differentiation for parameter learning.
    """
    
    def __init__(
        self,
        dt: float = 0.001,
    ):
        self.dt = dt
        
    def _spring_force(self, x: jnp.ndarray, v: jnp.ndarray, k: float, c: float) -> jnp.ndarray:
        """Spring-damper force: F = -k*x - c*v"""
        return -k * x - c * v
    
    def _step_symplectic(
        self,
        x: jnp.ndarray,
        v: jnp.ndarray,
        mass: float,
        stiffness: float,
        damping: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Symplectic (Verlet) integration step.
        
        Conserves energy better than naive Euler.
        """
        # Half-step velocity
        a_half = self._spring_force(x, v, stiffness, damping) / mass
        v_half = v + 0.5 * self.dt * a_half
        
        # Full-step position
        x_new = x + self.dt * v_half
        
        # Half-step velocity again
        a_full = self._spring_force(x_new, v_half, stiffness, damping) / mass
        v_new = v_half + 0.5 * self.dt * a_full
        
        return x_new, v_new
    
    def simulate(
        self,
        x0: jnp.ndarray,
        v0: jnp.ndarray,
        params: jnp.ndarray,  # [mass, stiffness, damping]
        n_steps: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate forward using JAX.
        
        Args:
            x0: Initial positions (shape: [3])
            v0: Initial velocities (shape: [3])
            params: [mass, stiffness, damping]
            n_steps: Number of simulation steps
        
        Returns:
            positions: Trajectory (shape: [n_steps, 3])
            velocities: Velocity trajectory (shape: [n_steps, 3])
        """
        mass = params[0]
        stiffness = params[1]
        damping = params[2]
        
        # Initialize trajectory
        positions = jnp.zeros((n_steps, 3))
        velocities = jnp.zeros((n_steps, 3))
        
        positions = positions.at[0].set(x0)
        velocities = velocities.at[0].set(v0)
        
        def step_fn(carry, t):
            x, v = carry
            x_new, v_new = self._step_symplectic(x, v, mass, stiffness, damping)
            return (x_new, v_new), (x_new, v_new)
        
        init_carry = (x0, v0)
        _, (pos_traj, vel_traj) = jax.lax.scan(step_fn, init_carry, jnp.arange(n_steps))
        
        return pos_traj, vel_traj
    
    def loss_fn(
        self,
        params: jnp.ndarray,
        x0: jnp.ndarray,
        v0: jnp.ndarray,
        x_obs: jnp.ndarray,
    ) -> float:
        """
        MSE loss between simulated and observed trajectories.
        
        This is what we'll differentiate with respect to params.
        """
        pos_traj, _ = self.simulate(x0, v0, params, len(x_obs))
        
        # MSE loss
        loss = jnp.mean((pos_traj - x_obs) ** 2)
        
        return loss
    
    def learn_params(
        self,
        x0: np.ndarray,
        v0: np.ndarray,
        x_obs: np.ndarray,
        init_params: Optional[np.ndarray] = None,
        lr: float = 0.1,
        n_iterations: int = 100,
    ) -> Tuple[np.ndarray, float]:
        """
        Learn physics parameters using gradient descent.
        
        Uses jax.grad to compute ∇L with respect to params.
        """
        if init_params is None:
            init_params = np.array([1.0, 100.0, 1.0])  # [mass, stiffness, damping]
        
        # Convert to JAX
        x0_jax = jnp.array(x0)
        v0_jax = jnp.array(v0)
        x_obs_jax = jnp.array(x_obs)
        
        # JIT-compile
        _loss_fn = jit(self.loss_fn)
        _grad_fn = jit(grad(self.loss_fn))
        
        params = init_params.astype(np.float32)
        best_loss = float('inf')
        best_params = params.copy()
        
        log.info(f"Starting JAX parameter learning: {n_iterations} iterations")
        
        for i in range(n_iterations):
            # Compute loss and gradient
            loss = float(_loss_fn(params, x0_jax, v0_jax, x_obs_jax))
            grads = np.array(_grad_fn(params, x0_jax, v0_jax, x_obs_jax))
            
            # Gradient descent step
            params = params - lr * grads
            
            # Ensure positive values
            params = np.maximum(params, [0.01, 0.1, 0.0])
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if i % 20 == 0:
                log.info(f"  Iter {i}: loss={loss:.6f}, "
                        f"m={params[0]:.4f}, k={params[1]:.4f}, c={params[2]:.4f}")
        
        log.info(f"JAX learned params: m={best_params[0]:.4f}, "
                f"k={best_params[1]:.4f}, c={best_params[2]:.4f}")
        
        return best_params, best_loss


class MuJoCoBridge:
    """Bridge between JAX differentiable physics and MuJoCo."""
    
    @staticmethod
    def learned_to_mujoco(
        learned_params: np.ndarray,
        trajectory: np.ndarray,
        mechanism_type: str = "rigid_body",
    ) -> str:
        """
        Convert learned JAX parameters to MuJoCo XML.
        """
        m, k, c = learned_params[0], learned_params[1], learned_params[2]
        
        # Get geometry from trajectory
        if len(trajectory) > 0:
            pos_min = trajectory.min(axis=0)
            pos_max = trajectory.max(axis=0)
            size = (pos_max - pos_min) / 2 + 0.05
            center = (pos_max + pos_min) / 2
        else:
            size = np.array([0.1, 0.1, 0.1])
            center = np.array([0.0, 0.5, 0.0])
        
        xml = f"""
<mujoco model="aether_jax_learned">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option integrator="implicitfast"/>
  
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.01" rgba=".3 .3 .3 1" friction="0.5 0.01 0.01"/>
    
    <body name="body1" pos="{center[0]:.4f} {center[1]:.4f} {center[2]:.4f}">
      <freejoint/>
      <inertial pos="0 0 0" mass="{m:.4f}" fullinertia="0.001 0.001 0.001 0 0 0"/>
      <geom type="box" size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" 
            rgba="0.2 0.8 0.8 1"/>
    </body>
  </worldbody>
  
  <!-- JAX Learned Parameters: -->
  <!-- mass={m:.4f}kg, stiffness={k:.2f}N/m, damping={c:.4f}Ns/m -->
</mujoco>"""
        
        return xml


def run_differentiable_physics_pipeline(
    observed_trajectory: np.ndarray,
    mechanism_type: str = "mass_spring_damper",
    lr: float = 0.1,
    n_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Complete pipeline: Learn physics from trajectory using JAX.
    """
    log.info(f"Running JAX differentiable physics pipeline on {len(observed_trajectory)} frames")
    
    # Initialize simulator
    simulator = DifferentiableSimulator(dt=1/30)  # 30 FPS
    
    # Initial conditions
    x0 = observed_trajectory[0]
    v0 = np.gradient(observed_trajectory, axis=0)[0]
    
    # Learn parameters
    learned_params, final_loss = simulator.learn_params(
        x0, v0, observed_trajectory,
        lr=lr,
        n_iterations=n_iterations,
    )
    
    # Generate MuJoCo model
    mj_xml = MuJoCoBridge.learned_to_mujoco(learned_params, observed_trajectory, mechanism_type)
    
    return {
        "learned_params": {
            "mass_kg": float(learned_params[0]),
            "stiffness_Nm": float(learned_params[1]),
            "damping_Nsm": float(learned_params[2]),
        },
        "final_loss": float(final_loss),
        "mujoco_xml": mj_xml,
        "mechanism_type": mechanism_type,
    }


def test_differentiable_physics():
    """Test the differentiable physics pipeline."""
    log.info("Testing JAX differentiable physics...")
    
    # Generate synthetic data: damped oscillation
    t = np.linspace(0, 2, 60)  # 2 seconds at 30 FPS
    omega = 2 * np.pi * 2  # 2 Hz natural frequency
    zeta = 0.1  # damping ratio
    
    x0_true = 0.1  # initial displacement
    m_true = 1.0
    k_true = m_true * omega**2
    c_true = 2 * zeta * np.sqrt(k_true * m_true)
    
    # Damped oscillation
    x_obs = x0_true * np.exp(-zeta * omega * t) * np.cos(omega * np.sqrt(1-zeta**2) * t)
    x_obs = np.stack([x_obs, np.zeros_like(x_obs), np.zeros_like(x_obs)], axis=1)
    
    # Add some noise
    np.random.seed(42)
    x_obs = x_obs + np.random.randn(*x_obs.shape) * 0.005
    
    # Learn parameters
    result = run_differentiable_physics_pipeline(x_obs, n_iterations=100)
    
    log.info(f"\nResults:")
    log.info(f"  True: m={m_true:.4f}, k={k_true:.4f}, c={c_true:.4f}")
    log.info(f"  Learned: m={result['learned_params']['mass_kg']:.4f}, "
            f"k={result['learned_params']['stiffness_Nm']:.4f}, "
            f"c={result['learned_params']['damping_Nsm']:.4f}")
    log.info(f"  Final loss: {result['final_loss']:.6f}")
    
    return result


if __name__ == "__main__":
    test_differentiable_physics()
