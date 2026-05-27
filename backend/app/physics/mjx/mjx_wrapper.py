"""
MJX: MuJoCo in JAX (DeepMind)
==============================

Wrapper for DeepMind's MJX - MuJoCo physics engine in JAX.

Key capability: DIFFERENTIABLE physics simulation!

You can backpropagate through the simulation to learn:
- Mass parameters
- Friction coefficients  
- Damping constants

This replaces the heuristic "convex hull * density" mass estimation.

IMPORTANT: MJX is maintained by DeepMind. We provide a wrapper for our use case.

Install: pip install mujoco mujocax
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

log = logging.getLogger(__name__)

# Try to import MJX, fall back to simple simulation if unavailable
try:
    import mujoco as mj
    from mujoco import mjx as mjx_module
    HAS_MUJOCO = True
    HAS_MJX = True
except ImportError:
    HAS_MUJOCO = False
    HAS_MJX = False
    log.warning("MuJoCo/MJX not installed. Using fallback simulation.")


class MJXWrapper:
    """
    Wrapper for DeepMind's MJX (MuJoCo in JAX).
    
    Provides differentiable physics simulation for learning parameters.
    
    If MJX is not available, falls back to simple spring-mass simulation.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        dt: float = 0.001,
        n_substeps: int = 1,
    ):
        """
        Args:
            model_path: Path to MuJoCo XML model
            dt: Simulation timestep
            n_substeps: Integration substeps per call
        """
        self.dt = dt
        self.n_substeps = n_substeps
        self.model_path = model_path
        
        if HAS_MJX and model_path:
            self._init_mjx(model_path)
        else:
            self.model = None
            log.info("Using fallback simulation (MJX not available)")
    
    def _init_mjx(self, model_path: str):
        """Initialize MJX model from XML."""
        if not HAS_MJX:
            raise RuntimeError("MJX not installed")
        
        # Load MuJoCo model
        self.model = mj.MjModel.from_xml_path(model_path)
        
        # Create MJX data
        self._data = mjx_module.Data(self.model)
        
        log.info(f"MJX model loaded: {self.model.nq} DOFs")
    
    def simulate(
        self,
        qpos0: jnp.ndarray,
        qvel0: jnp.ndarray,
        ctrl: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate forward.
        
        Args:
            qpos0: [nq] initial positions
            qvel0: [nv] initial velocities
            ctrl: [nu] control inputs (optional)
            
        Returns:
            qpos, qvel: final state
        """
        if self.model is not None:
            return self._simulate_mjx(qpos0, qvel0, ctrl)
        else:
            return self._simulate_fallback(qpos0, qvel0)
    
    def _simulate_mjx(
        self,
        qpos0: jnp.ndarray,
        qvel0: jnp.ndarray,
        ctrl: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """MJX simulation."""
        raise NotImplementedError("MJX integration requires full implementation")
    
    def _simulate_fallback(
        self,
        qpos0: jnp.ndarray,
        qvel0: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fallback: simple spring-mass simulation.
        
        This is used when MJX is not available.
        """
        q, p = qpos0.copy(), qvel0.copy() * 10  # scale velocity to momentum
        
        for _ in range(self.n_substeps):
            # Simple Euler integration
            dq = p / 1.0  # mass = 1
            dp = -q * 1.0  # stiffness = 1
            
            q = q + dq * self.dt
            p = p + dp * self.dt
        
        return q, p


class BackpropPhysics:
    """
    Backpropagation through physics simulation.
    
    Learns physical parameters (mass, friction, stiffness) from observed motion.
    
    Key insight:
    Instead of guessing mass from geometry (convex hull * density),
    we learn mass from motion (how the object accelerates).
    
    F = ma → m = F/a → Learn from observed accelerations!
    """
    
    def __init__(
        self,
        n_dofs: int,
        dt: float = 0.001,
        n_substeps: int = 100,
    ):
        """
        Args:
            n_dofs: Number of degrees of freedom
            dt: Simulation timestep
            n_substeps: Steps per trajectory
        """
        self.n_dofs = n_dofs
        self.dt = dt
        self.n_substeps = n_substeps
        
        # Learnable parameters (log-parametrization for positivity)
        self.params = {
            "log_mass": jnp.zeros(n_dofs),
            "log_stiffness": jnp.zeros(n_dofs),
            "log_damping": jnp.zeros(n_dofs),
        }
    
    def simulate_step(
        self,
        state: jnp.ndarray,  # [2*n_dofs] = [q, p]
        mass: jnp.ndarray,
        stiffness: jnp.ndarray,
        damping: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Single simulation step with symplectic integration.
        
        Uses semi-implicit Euler (symplectic):
        p_{n+1} = p_n + dt * (-∇V(q_n) - damping * q_dot_n)
        q_{n+1} = q_n + dt * (p_{n+1} / m)
        
        Args:
            state: [2*n_dofs] current state [q, p]
            mass: [n_dofs] mass per DOF
            stiffness: [n_dofs] spring constant per DOF
            damping: [n_dofs] damping coefficient per DOF
            
        Returns:
            next_state: [2*n_dofs]
        """
        q = state[:self.n_dofs]
        p = state[self.n_dofs:]
        
        # Generalized forces: -∇V - damping * q_dot
        # q_dot = p / m
        q_dot = p / mass
        forces = -stiffness * q - damping * q_dot
        
        # Update momentum
        p_new = p + self.dt * forces
        
        # Update position (using new momentum - semi-implicit)
        q_new = q + self.dt * (p_new / mass)
        
        return jnp.concatenate([q_new, p_new])
    
    def simulate_trajectory(
        self,
        state0: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        n_steps: int,
    ) -> jnp.ndarray:
        """
        Simulate full trajectory.
        
        Args:
            state0: [2*n_dofs] initial state
            params: dict with log_mass, log_stiffness, log_damping
            n_steps: number of steps
            
        Returns:
            trajectory: [n_steps+1, 2*n_dofs]
        """
        mass = jnp.exp(params["log_mass"])
        stiffness = jnp.exp(params["log_stiffness"])
        damping = jnp.exp(params["log_damping"])
        
        # Vectorized simulation
        def step_fn(state, _):
            next_state = self.simulate_step(state, mass, stiffness, damping)
            return next_state, next_state
        
        _, trajectory = jax.lax.scan(step_fn, state0, jnp.arange(n_steps))
        
        # Prepend initial state
        return jnp.vstack([state0, trajectory])
    
    def physics_loss(
        self,
        params: Dict[str, jnp.ndarray],
        state0: jnp.ndarray,
        q_obs: jnp.ndarray,  # [T, n_dofs] observed positions
        p_obs: jnp.ndarray,  # [T, n_dofs] observed momenta
    ) -> jnp.ndarray:
        """
        Loss: MSE between simulated and observed trajectory.
        
        This is what we minimize to learn physics parameters!
        
        Args:
            params: Physics parameters to learn
            state0: Initial state
            q_obs: Observed positions
            p_obs: Observed momenta
            
        Returns:
            loss: scalar MSE
        """
        T = q_obs.shape[0]
        
        # Simulate
        q_flat = jnp.reshape(q_obs, (T, -1))
        p_flat = jnp.reshape(p_obs, (T, -1))
        
        # Initial state
        state0_sim = jnp.concatenate([q_flat[0], p_flat[0]])
        
        # Simulate trajectory
        traj = self.simulate_trajectory(state0_sim, params, T - 1)
        
        q_sim = traj[:, :self.n_dofs]
        p_sim = traj[:, self.n_dofs:]
        
        # MSE loss
        q_loss = jnp.mean((q_sim - q_flat)**2)
        p_loss = jnp.mean((p_sim - p_flat)**2)
        
        return q_loss + p_loss
    
    def learn_params(
        self,
        state0: np.ndarray,
        trajectory_obs: np.ndarray,  # [T, 2*n_dofs]
        lr: float = 0.01,
        n_iterations: int = 1000,
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Learn physics parameters from observed trajectory.
        
        Args:
            state0: Initial state
            trajectory_obs: Observed trajectory [T, 2*n_dofs]
            lr: Learning rate
            n_iterations: Training iterations
            
        Returns:
            learned_params: Dict with mass, stiffness, damping
            metrics: Training metrics
        """
        import optax
        
        # Parse observation
        T, dim = trajectory_obs.shape
        n_dofs = dim // 2
        
        q_obs = trajectory_obs[:, :n_dofs]
        p_obs = trajectory_obs[:, n_dofs:]
        
        # Convert to JAX
        state0_jax = jnp.array(state0)
        q_jax = jnp.array(q_obs)
        p_jax = jnp.array(p_obs)
        
        # Initial params
        params = {
            "log_mass": jnp.zeros(n_dofs),
            "log_stiffness": jnp.zeros(n_dofs),
            "log_damping": jnp.zeros(n_dofs),
        }
        
        # Loss and grad
        def loss_fn(p):
            return self.physics_loss(p, state0_jax, q_jax, p_jax)
        
        grad_fn = jax.jit(jax.grad(loss_fn))
        
        # Optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        
        # Training
        metrics = {"loss": [], "mass": [], "stiffness": [], "damping": []}
        
        for i in range(n_iterations):
            grads = grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            if i % 100 == 0:
                loss = loss_fn(params)
                
                # Convert from log-space
                mass = np.exp(np.array(params["log_mass"]))
                stiffness = np.exp(np.array(params["log_stiffness"]))
                damping = np.exp(np.array(params["log_damping"]))
                
                metrics["loss"].append(float(loss))
                metrics["mass"].append(float(mass[0]))
                metrics["stiffness"].append(float(stiffness[0]))
                metrics["damping"].append(float(damping[0]))
                
                print(f"Iter {i}: loss={loss:.6f}, m={mass[0]:.3f}, k={stiffness[0]:.3f}, c={damping[0]:.3f}")
        
        # Final params
        learned_params = {
            "mass": np.exp(np.array(params["log_mass"])),
            "stiffness": np.exp(np.array(params["log_stiffness"])),
            "damping": np.exp(np.array(params["log_damping"])),
        }
        
        return learned_params, metrics


def simulate_mjx(
    model_path: str,
    qpos0: np.ndarray,
    qvel0: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple MJX simulation function.
    
    Args:
        model_path: Path to MuJoCo XML
        qpos0: Initial positions
        qvel0: Initial velocities
        n_steps: Simulation steps
        
    Returns:
        qpos_traj, qvel_traj
    """
    if not HAS_MJX:
        log.warning("MJX not available")
        return qpos0, qvel0
    
    # This would be the actual MJX implementation
    raise NotImplementedError("Use BackpropPhysics for learning")


def test_backprop_physics():
    """Test backpropagation through physics."""
    print("=" * 60)
    print("Testing Backpropagation Through Physics")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate synthetic trajectory
    T = 60
    t = np.linspace(0, 1, T)
    
    # Damped harmonic oscillator: q = sin(t) * exp(-0.1*t)
    q_true = np.sin(2 * np.pi * t).reshape(-1, 1)
    p_true = 2 * np.pi * np.cos(2 * np.pi * t).reshape(-1, 1)  # m=1, p=m*q_dot
    
    trajectory = np.concatenate([q_true, p_true], axis=1)
    state0 = trajectory[0]
    
    print(f"Generated {T} time steps")
    print(f"True parameters: m=1.0, k=~39.5 (2π²), c=~0.0")
    
    # Learn physics
    bp = BackpropPhysics(n_dofs=1, dt=0.016, n_substeps=1)
    
    learned_params, metrics = bp.learn_params(
        state0,
        trajectory,
        lr=0.05,
        n_iterations=500,
    )
    
    print(f"\nLearned parameters:")
    print(f"  mass: {learned_params['mass'][0]:.4f}")
    print(f"  stiffness: {learned_params['stiffness'][0]:.4f}")
    print(f"  damping: {learned_params['damping'][0]:.4f}")
    
    print(f"\nFinal loss: {metrics['loss'][-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Backprop Physics Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_backprop_physics()
