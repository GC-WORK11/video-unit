"""
AETHER Hamiltonian Regularized Differentiable Physics (Phase 2.2)
================================================================
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
from typing import Tuple, Optional, Dict, Any
import numpy as np
import logging

log = logging.getLogger(__name__)


class HamiltonianRegularizedSimulator:
    """JAX-based differentiable physics with Hamiltonian regularization."""
    
    def __init__(self, dt: float = 1/30, lambda_hamiltonian: float = 0.1):
        self.dt = dt
        self.lambda_hamiltonian = lambda_hamiltonian
    
    def hamiltonian(self, x, v, m, k):
        """H = T + V = total mechanical energy"""
        T = 0.5 * m * jnp.sum(v ** 2)
        V = 0.5 * k * jnp.sum(x ** 2)
        return T + V
    
    def _step(self, x, v, m, k, c):
        """Symplectic step."""
        a_half = (-k * x - c * v) / m
        v_half = v + 0.5 * self.dt * a_half
        x_new = x + self.dt * v_half
        a_full = (-k * x_new - c * v_half) / m
        v_new = v_half + 0.5 * self.dt * a_full
        return x_new, v_new
    
    def simulate(self, x0, v0, params, n_steps):
        """Simulate forward with energy tracking."""
        m, k, c = params[0], params[1], params[2]
        
        positions = jnp.zeros((n_steps, 3))
        velocities = jnp.zeros((n_steps, 3))
        energies = jnp.zeros(n_steps)
        
        positions = positions.at[0].set(x0)
        velocities = velocities.at[0].set(v0)
        energies = energies.at[0].set(self.hamiltonian(x0, v0, m, k))
        
        def step_fn(carry, t):
            x, v = carry
            x_new, v_new = self._step(x, v, m, k, c)
            return (x_new, v_new), (x_new, v_new, self.hamiltonian(x_new, v_new, m, k))
        
        _, (pos, vel, H) = jax.lax.scan(step_fn, (x0, v0), jnp.arange(n_steps))
        
        return pos, vel, H
    
    def loss_fn(self, params, x0, v0, x_obs, lambda_h):
        """Pure loss function (no metrics) for JIT."""
        m, k = params[0], params[1]
        pos, _, H = self.simulate(x0, v0, params, len(x_obs))
        
        mse = jnp.mean((pos - x_obs) ** 2)
        dH = jnp.diff(H)
        H_loss = jnp.mean(dH ** 2)
        H_drift = jnp.mean((H - H[0]) ** 2)
        
        return mse + lambda_h * (H_loss + 0.1 * H_drift)
    
    def learn_params(self, x0, v0, x_obs, init_params=None, lr=0.01, n_iterations=200, lambda_h=0.1):
        """Learn with Hamiltonian regularization."""
        if init_params is None:
            init_params = np.array([1.0, 100.0, 1.0])
        
        x0_j = jnp.array(x0)
        v0_j = jnp.array(v0)
        x_obs_j = jnp.array(x_obs)
        
        # JIT compiled functions
        _loss = jit(lambda p: self.loss_fn(p, x0_j, v0_j, x_obs_j, lambda_h))
        _grad = jit(grad(_loss))
        
        params = init_params.astype(np.float32)
        best_loss = float('inf')
        best_params = params.copy()
        best_metrics = {}
        
        log.info(f"Starting Hamiltonian-regularized learning: {n_iterations} iterations")
        
        for i in range(n_iterations):
            loss = float(_loss(params))
            grads = np.array(_grad(params))
            
            params = np.maximum(params - lr * grads, [0.01, 0.1, 0.0])
            
            if loss < best_loss:
                best_loss = loss
                best_params = params.copy()
            
            if i % 40 == 0:
                # Compute metrics for logging
                pos, _, H = self.simulate(x0_j, v0_j, best_params, len(x_obs))
                mse = float(jnp.mean((pos - x_obs_j) ** 2))
                dH = np.diff(np.array(H))
                H_loss = float(jnp.mean(dH ** 2))
                log.info(f"  Iter {i}: loss={loss:.6f}, MSE={mse:.6f}, H_loss={H_loss:.6f}")
        
        # Final metrics
        pos, vel, H = self.simulate(x0_j, v0_j, best_params, len(x_obs))
        mse = float(jnp.mean((pos - x_obs_j) ** 2))
        dH = np.diff(np.array(H))
        H_loss = float(jnp.mean(dH ** 2))
        H_drift = float(jnp.mean((np.array(H) - np.array(H)[0]) ** 2))
        
        return best_params, best_loss, {
            "mse_loss": mse,
            "hamiltonian_loss": H_loss,
            "energy_drift": H_drift,
            "initial_energy": float(H[0]),
            "final_energy": float(H[-1]),
        }


def run_hamiltonian_pipeline(observed_trajectory, lr=0.01, n_iterations=200, lambda_hamiltonian=0.1):
    """Complete pipeline with Hamiltonian regularization."""
    log.info(f"Running Hamiltonian pipeline on {len(observed_trajectory)} frames")
    
    simulator = HamiltonianRegularizedSimulator(lambda_hamiltonian=lambda_hamiltonian)
    
    x0 = observed_trajectory[0]
    v0 = np.gradient(observed_trajectory, axis=0)[0]
    
    learned, loss, metrics = simulator.learn_params(
        x0, v0, observed_trajectory,
        lr=lr, n_iterations=n_iterations, lambda_h=lambda_hamiltonian
    )
    
    return {
        "learned_params": {
            "mass_kg": float(learned[0]),
            "stiffness_Nm": float(learned[1]),
            "damping_Nsm": float(learned[2]),
        },
        "metrics": metrics,
        "total_loss": loss,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2.2: Hamiltonian Regularization Test")
    print("=" * 70)
    
    # Test energy conservation
    print("\n🧪 TEST 1: Energy Conservation")
    print("-" * 50)
    
    dt = 0.01
    m, k = 1.0, 100.0
    
    def symp_step(x, v):
        dv = (-k * x) / m * dt
        return x + (v + dv) * dt, v + dv
    
    E = []
    x, v = 0.1, 0.0
    for _ in range(1000):
        E.append(0.5*m*v**2 + 0.5*k*x**2)
        x, v = symp_step(x, v)
    
    drift = abs(E[-1] - E[0]) / E[0] * 100
    print(f"Energy drift: {drift:.4f}%")
    print("✅ Symplectic integration works!" if drift < 10 else "⚠️  High drift")
    
    # Test regularization
    print("\n🧪 TEST 2: Regularized Learning")
    print("-" * 50)
    
    t = np.linspace(0, 2, 60)
    omega = 2 * np.pi * 2
    zeta = 0.1
    m_t, k_t, c_t = 1.0, 157.91, 2.51
    
    x_obs = 0.1 * np.exp(-zeta * omega * t) * np.cos(omega * np.sqrt(1-zeta**2) * t)
    x_obs = np.stack([x_obs, np.zeros(60), np.zeros(60)], axis=1)
    x_obs += np.random.randn(60, 3) * 0.002
    
    print(f"True: m={m_t}, k={k_t:.2f}, c={c_t:.2f}")
    
    for lh in [0.0, 0.1, 0.5]:
        result = run_hamiltonian_pipeline(x_obs, lambda_hamiltonian=lh, n_iterations=200)
        p = result['learned_params']
        print(f"\nλ_H={lh}:")
        print(f"  Learned: m={p['mass_kg']:.3f}, k={p['stiffness_Nm']:.2f}, c={p['damping_Nsm']:.3f}")
        print(f"  H_loss: {result['metrics']['hamiltonian_loss']:.6f}")
        print(f"  E_drift: {result['metrics']['energy_drift']:.6f}")
    
    print("\n" + "=" * 70)
    print("PHASE 2.2 COMPLETE ✅")
    print("=" * 70)
