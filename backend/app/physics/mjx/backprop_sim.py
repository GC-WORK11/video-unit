"""
AETHER Differentiable MJX System Identification
===============================================

This module implements TRUE backpropagation through the MuJoCo engine
using DeepMind's MJX.

Instead of guessing mass from geometry, we learn mass, friction, and
stiffness by backpropagating through the real physics equations of motion.
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import mujoco
from mujoco import mjx
import numpy as np
import optax
from typing import Tuple, Dict, Any, Optional
import logging

log = logging.getLogger(__name__)


class BackpropMJX:
    """
    Learns physical parameters by backpropagating through MJX.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to MuJoCo XML/MJCF model
        """
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.mjx_model = mjx.put_model(self.model)
        self.dt = self.model.opt.timestep
        
    @jit
    def simulate_step(self, model: mjx.Model, data: mjx.Data) -> mjx.Data:
        """Single step of MJX simulation."""
        return mjx.step(model, data)
    
    def simulate_trajectory(
        self,
        model: mjx.Model,
        qpos0: jnp.ndarray,
        qvel0: jnp.ndarray,
        ctrls: Optional[jnp.ndarray],
        n_steps: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate a full trajectory using lax.scan.
        """
        data = mjx.make_data(model)
        data = data.replace(qpos=qpos0, qvel=qvel0)
        
        if ctrls is None:
            ctrls = jnp.zeros((n_steps, model.nu))
            
        def step_fn(d, ctrl):
            d = d.replace(ctrl=ctrl)
            d = mjx.step(model, d)
            return d, (d.qpos, d.qvel)
        
        _, (qpos_traj, qvel_traj) = jax.lax.scan(step_fn, data, ctrls)
        
        return jnp.vstack([qpos0, qpos_traj]), jnp.vstack([qvel0, qvel_traj])

    def loss_fn(
        self,
        log_params: Dict[str, jnp.ndarray],
        qpos0: jnp.ndarray,
        qvel0: jnp.ndarray,
        qpos_obs: jnp.ndarray,
        ctrls: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """MSE loss between MJX simulation and observed positions."""
        curr_model = self.mjx_model
        
        if "log_body_mass" in log_params:
            curr_model = curr_model.replace(
                body_mass=jnp.exp(log_params["log_body_mass"])
            )
            
        if "log_geom_friction" in log_params:
            curr_model = curr_model.replace(
                geom_friction=jnp.exp(log_params["log_geom_friction"])
            )
            
        n_steps = qpos_obs.shape[0] - 1
        q_sim, _ = self.simulate_trajectory(curr_model, qpos0, qvel0, ctrls, n_steps)
        
        return jnp.mean((q_sim - qpos_obs) ** 2)

    def learn_parameters(
        self,
        qpos_obs: np.ndarray,
        qvel_obs: Optional[np.ndarray] = None,
        ctrls: Optional[np.ndarray] = None,
        lr: float = 0.05,
        n_iterations: int = 500,
        target_params: list[str] = ["body_mass"],
    ) -> Dict[str, Any]:
        """Run the optimization loop to discover physical parameters."""
        qpos_jax = jnp.array(qpos_obs)
        qpos0 = qpos_jax[0]
        
        if qvel_obs is not None:
            qvel0 = jnp.array(qvel_obs[0])
        else:
            qvel0 = (qpos_jax[1] - qpos_jax[0]) / self.dt
            
        initial_log_params = {}
        if "body_mass" in target_params:
            initial_log_params["log_body_mass"] = jnp.log(self.mjx_model.body_mass)
        if "geom_friction" in target_params:
            initial_log_params["log_geom_friction"] = jnp.log(self.mjx_model.geom_friction)
            
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(initial_log_params)
        
        _loss_fn = jit(lambda p: self.loss_fn(p, qpos0, qvel0, qpos_jax, ctrls))
        _grad_fn = jit(grad(_loss_fn))
        
        log_params = initial_log_params
        history = []
        
        for i in range(n_iterations):
            loss = _loss_fn(log_params)
            grads = _grad_fn(log_params)
            updates, opt_state = optimizer.update(grads, opt_state)
            log_params = optax.apply_updates(log_params, updates)
            
            if i % 50 == 0:
                history.append(float(loss))
                log.info(f"  Iteration {i}: loss = {loss:.10f}")
                
        learned = {k.replace("log_", ""): np.exp(np.array(v)) for k, v in log_params.items()}
        
        return {
            "learned_parameters": learned,
            "final_loss": float(_loss_fn(log_params)),
            "history": history,
        }


def run_test():
    """Definitive MJX System ID test."""
    import tempfile
    import os
    
    mjcf_xml = """
<mujoco>
  <option timestep="0.01" />
  <worldbody>
    <light pos="0 0 3" />
    <body name="pendulum" pos="0 0 1">
      <joint name="joint" type="hinge" axis="0 1 0" />
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5" mass="1.0" />
    </body>
  </worldbody>
</mujoco>
"""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w") as f:
        f.write(mjcf_xml)
        model_path = f.name
        
    try:
        sysid = BackpropMJX(model_path)
        
        # Ground truth: mass=2.0
        gt_model = sysid.mjx_model.replace(body_mass=sysid.mjx_model.body_mass.at[1].set(2.0))
        q0 = jnp.array([0.5])
        v0 = jnp.array([0.0])
        q_obs, q_vel_obs = sysid.simulate_trajectory(gt_model, q0, v0, None, 50)
        
        # Learning with true velocity
        print("\n--- Starting MJX System ID (Ground Truth v0) ---")
        result = sysid.learn_parameters(
            np.array(q_obs), 
            qvel_obs=np.array(q_vel_obs), 
            n_iterations=500, 
            lr=0.05
        )
        
        learned_mass = result['learned_parameters']['body_mass'][1]
        print(f"\nTrue mass: 2.0 | Learned: {learned_mass:.6f}")
        print(f"Final Loss: {result['final_loss']:.12f}")
        
        if abs(learned_mass - 2.0) < 1e-4:
            print("✅ MJX System ID SUCCESS (Mathematical Integrity Verified)")
        else:
            print("❌ MJX System ID FAILED (Check gradients or local minima)")
            
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_test()
