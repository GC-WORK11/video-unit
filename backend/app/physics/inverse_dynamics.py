"""
AETHER Differentiable Inverse Dynamics Engine (Phase 2)
======================================================

Learn physical parameters (mass, stiffness, damping) using gradient descent 
on a differentiable physics model.

THE PHYSICS:
- Mass-Spring-Damper ODE: m*x'' + c*x' + k*x = 0
- Parameters to learn: m, c, k
- Method: Gradient descent (Adam) minimizing MSE between simulated and observed trajectory.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)

class DifferentiableSpringDamper(nn.Module):
    """A differentiable mass-spring-damper system."""
    def __init__(self, init_m=1.0, init_k=100.0, init_c=1.0):
        super().__init__()
        # Parameters are stored in log-space to ensure they stay positive
        self.log_m = nn.Parameter(torch.tensor(np.log(init_m), dtype=torch.float32))
        self.log_k = nn.Parameter(torch.tensor(np.log(init_k), dtype=torch.float32))
        self.log_c = nn.Parameter(torch.tensor(np.log(init_c), dtype=torch.float32))

    @property
    def m(self): return torch.exp(self.log_m)
    @property
    def k(self): return torch.exp(self.log_k)
    @property
    def c(self): return torch.exp(self.log_c)

    def forward(self, x0, v0, t_steps):
        """
        Simulate the system using Euler integration (differentiable).
        
        Args:
            x0: Initial position (scalar)
            v0: Initial velocity (scalar)
            t_steps: Number of steps
            dt: Time step size
        """
        dt = t_steps[1] - t_steps[0]
        x = x0
        v = v0
        
        positions = [x]
        
        for _ in range(len(t_steps) - 1):
            # F = -k*x - c*v
            # a = F/m
            a = (-self.k * x - self.c * v) / self.m
            v = v + a * dt
            x = x + v * dt
            positions.append(x)
            
        return torch.stack(positions)

def learn_from_trajectory(
    traj: np.ndarray, 
    fps: float = 30.0,
    epochs: int = 100,
    lr: float = 0.1
) -> Dict[str, Any]:
    """
    Learn physics parameters from a trajectory using gradient descent.
    
    Args:
        traj: (N, 3) or (N,) array of positions
        fps: Frames per second
    """
    # 1. Preprocess trajectory
    if traj.ndim == 2:
        # Use the axis with most motion
        ranges = np.ptp(traj, axis=0)
        axis = np.argmax(ranges)
        signal = traj[:, axis]
    else:
        signal = traj
        
    signal = signal - np.mean(signal) # Center it
    n = len(signal)
    dt = 1.0 / fps
    t = torch.linspace(0, (n-1)*dt, n)
    obs = torch.tensor(signal, dtype=torch.float32)
    
    # 2. Initialize model
    # Heuristic initials
    init_m = 1.0
    init_k = 100.0 
    init_c = 5.0
    
    model = DifferentiableSpringDamper(init_m, init_k, init_c)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initial state
    x0 = obs[0]
    v0 = (obs[1] - obs[0]) / dt
    
    # 3. Optimization loop
    best_loss = float('inf')
    best_params = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred = model(x0, v0, t)
        loss = nn.MSELoss()(pred, obs)
        
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                "mass_kg": model.m.item(),
                "stiffness_Nm": model.k.item(),
                "damping_Nsm": model.c.item(),
            }
            
    # 4. Calculate derived metrics
    m, k, c = best_params["mass_kg"], best_params["stiffness_Nm"], best_params["damping_Nsm"]
    omega_n = np.sqrt(k/m)
    f_n = omega_n / (2 * np.pi)
    zeta = c / (2 * np.sqrt(k*m))
    
    return {
        "natural_freq_Hz": float(f_n),
        "damping_ratio": float(zeta),
        "mass_kg": float(m),
        "stiffness_Nm": float(k),
        "damping_Nsm": float(c),
        "loss": float(best_loss),
        "confidence": float(max(0, 1 - best_loss / (np.var(signal) + 1e-6))),
        "method": "differentiable_physics_adam"
    }

def map_to_simulator_params(learned: dict, mechanism_type: str) -> dict:
    """Map learned physics params to universal simulator params."""
    # Simplified mapping
    mapped = {}
    if mechanism_type == "vehicle":
        mapped["suspension_stiffness"] = learned.get("stiffness_Nm", 10000)
        mapped["suspension_damping"] = learned.get("damping_Nsm", 1000)
        mapped["chassis_mass"] = learned.get("mass_kg", 50)
    elif mechanism_type == "pendulum":
        mapped["bob_mass"] = learned.get("mass_kg", 1.0)
    else:
        mapped["mass"] = learned.get("mass_kg", 1.0)
        
    return mapped
