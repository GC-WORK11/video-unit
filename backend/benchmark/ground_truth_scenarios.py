"""
Ground Truth Scenarios for AETHER Accuracy Benchmark
=====================================================

This module defines controlled physics scenarios with known ground truth parameters.
Each scenario generates synthetic trajectory data that AETHER must analyze.

The scenarios test:
1. Calibrated pendulum - tests mass, length, gravity, and period estimation
2. 2-link robot arm - tests multi-body mass and length estimation
3. Falling object with drag - tests mass and drag coefficient estimation
4. Spring-mass system - tests stiffness and damping estimation
5. Belt-gantry mechanism - tests linear motion parameters

References:
- Barsukov, "Robot Dynamics"
- Goldstein, "Classical Mechanics" (3rd ed)
- Featherstone, "Rigid Body Dynamics Algorithms"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum


class ScenarioType(str, Enum):
    PENDULUM = "pendulum"
    ROBOT_ARM = "robot_arm"
    FALLING_OBJECT = "falling_object"
    SPRING_MASS = "spring_mass"
    BELT_GANTRY = "belt_gantry"


@dataclass(frozen=True)
class GroundTruthParams:
    """Immutable ground truth parameters for a scenario."""
    masses_kg: Tuple[float, ...]
    lengths_m: Tuple[float, ...]
    friction_coef: float
    damping: float
    gravity_mps2: float = 9.81
    stiffness: Optional[float] = None
    drag_coef: Optional[float] = None
    area_m2: Optional[float] = None
    initial_angles: Optional[Tuple[float, ...]] = None
    initial_velocities: Optional[Tuple[float, ...]] = None


@dataclass
class TrajectoryData:
    """Simulated trajectory data from ground truth physics."""
    time_s: np.ndarray
    positions_3d: np.ndarray  # Shape: (n_timesteps, 3) or list of such arrays
    angles: np.ndarray  # Shape: (n_timesteps, n_joints)
    angular_velocities: np.ndarray  # Shape: (n_timesteps, n_joints)
    joint_forces: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ScenarioResult:
    """Result of running physics extraction on a scenario."""
    scenario_name: str
    scenario_type: ScenarioType
    ground_truth: GroundTruthParams
    extracted_params: Dict[str, float]
    period_s: Optional[float] = None
    extracted_period_s: Optional[float] = None
    trajectory_data: Optional[TrajectoryData] = None


def generate_pendulum_trajectory(
    mass_kg: float = 1.0,
    rod_length_m: float = 0.5,
    gravity_mps2: float = 9.81,
    damping: float = 0.0,
    initial_angle_rad: float = 0.5,
    duration_s: float = 10.0,
    dt_s: float = 0.001,
) -> TrajectoryData:
    """
    Generate pendulum trajectory using exact nonlinear equations.

    Physics:
        theta'' = -(g/L) * sin(theta) - c * theta'

    where:
        g = gravity
        L = rod length
        c = damping coefficient

    Period (small angle): T = 2*pi*sqrt(L/g)
    Small angle approximation error < 1% for angles < 0.3 rad

    Args:
        mass_kg: Mass of pendulum bob
        rod_length_m: Length of massless rod
        gravity_mps2: Gravitational acceleration
        damping: Linear damping coefficient at pivot
        initial_angle_rad: Initial angle from vertical
        duration_s: Simulation duration
        dt_s: Integration timestep

    Returns:
        TrajectoryData with full trajectory information
    """
    n_steps = min(int(duration_s / dt_s), 50000)
    dt = duration_s / n_steps

    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    t = np.zeros(n_steps)

    theta[0] = initial_angle_rad

    # Integrate using symplectic Euler for energy conservation
    for i in range(1, n_steps):
        alpha = -(gravity_mps2 / rod_length_m) * np.sin(theta[i-1]) - damping * omega[i-1]
        omega[i] = omega[i-1] + alpha * dt
        theta[i] = theta[i-1] + omega[i] * dt
        t[i] = t[i-1] + dt

    # Cartesian position of bob (x, y, z)
    x = rod_length_m * np.sin(theta)
    y = -rod_length_m * np.cos(theta)
    z = np.zeros_like(x)

    positions_3d = np.stack([x, y, z], axis=1)

    # Analytical period for comparison (small angle)
    period_s = 2 * np.pi * np.sqrt(rod_length_m / gravity_mps2)

    return TrajectoryData(
        time_s=t,
        positions_3d=positions_3d,
        angles=theta.reshape(-1, 1),
        angular_velocities=omega.reshape(-1, 1),
        metadata={
            "period_s": period_s,
            "dt_s": dt,
            "n_steps": n_steps,
        }
    )


def generate_2link_arm_trajectory(
    link1_mass_kg: float = 1.0,
    link2_mass_kg: float = 0.5,
    link1_length_m: float = 0.3,
    link2_length_m: float = 0.25,
    gravity_mps2: float = 9.81,
    joint1_damping: float = 0.0,
    joint2_damping: float = 0.0,
    initial_angle1_rad: float = 0.5,
    initial_angle2_rad: float = 0.3,
    duration_s: float = 10.0,
    dt_s: float = 0.001,
) -> TrajectoryData:
    """
    Generate 2-link robot arm trajectory using Euler-Lagrange equations.

    For a 2-link planar arm with point masses at joints:

    M(q)q'' + C(q,q')q' + g(q) = tau

    where M is the inertia matrix, C is Coriolis/centrifugal, g is gravity.

    Simplified decoupled equations (valid when coupling is weak):
        theta1'' = -(g/L1)*sin(theta1) - c1*theta1'
        theta2'' = -(g/L2)*sin(theta2) - c2*theta2'

    End effector position:
        x = L1*sin(theta1) + L2*sin(theta2)
        y = -L1*cos(theta1) - L2*cos(theta2)

    Args:
        link1_mass_kg: Mass of link 1
        link2_mass_kg: Mass of link 2
        link1_length_m: Length of link 1
        link2_length_m: Length of link 2
        gravity_mps2: Gravitational acceleration
        joint1_damping: Damping at joint 1
        joint2_damping: Damping at joint 2
        initial_angle1_rad: Initial angle of joint 1
        initial_angle2_rad: Initial angle of joint 2
        duration_s: Simulation duration
        dt_s: Integration timestep

    Returns:
        TrajectoryData with full arm trajectory
    """
    n_steps = min(int(duration_s / dt_s), 50000)
    dt = duration_s / n_steps

    theta1 = np.zeros(n_steps)
    theta2 = np.zeros(n_steps)
    omega1 = np.zeros(n_steps)
    omega2 = np.zeros(n_steps)
    t = np.zeros(n_steps)

    theta1[0] = initial_angle1_rad
    theta2[0] = initial_angle2_rad

    for i in range(1, n_steps):
        # Simplified Euler-Lagrange (decoupled)
        alpha1 = -(gravity_mps2 / link1_length_m) * np.sin(theta1[i-1]) - joint1_damping * omega1[i-1]
        alpha2 = -(gravity_mps2 / link2_length_m) * np.sin(theta2[i-1]) - joint2_damping * omega2[i-1]

        omega1[i] = omega1[i-1] + alpha1 * dt
        omega2[i] = omega2[i-1] + alpha2 * dt
        theta1[i] = theta1[i-1] + omega1[i] * dt
        theta2[i] = theta2[i-1] + omega2[i] * dt
        t[i] = t[i-1] + dt

    # End effector position
    x = link1_length_m * np.sin(theta1) + link2_length_m * np.sin(theta2)
    y = -link1_length_m * np.cos(theta1) - link2_length_m * np.cos(theta2)
    z = np.zeros_like(x)

    positions_3d = np.stack([x, y, z], axis=1)
    angles = np.stack([theta1, theta2], axis=1)
    angular_velocities = np.stack([omega1, omega2], axis=1)

    return TrajectoryData(
        time_s=t,
        positions_3d=positions_3d,
        angles=angles,
        angular_velocities=angular_velocities,
        metadata={
            "n_links": 2,
            "link1_length_m": link1_length_m,
            "link2_length_m": link2_length_m,
            "link1_mass_kg": link1_mass_kg,
            "link2_mass_kg": link2_mass_kg,
            "dt_s": dt,
            "n_steps": n_steps,
        }
    )


def generate_falling_object_trajectory(
    mass_kg: float = 1.0,
    drag_coef: float = 0.1,  # Drag coefficient Cd
    area_m2: float = 0.01,   # Frontal area
    gravity_mps2: float = 9.81,
    initial_height_m: float = 10.0,
    initial_velocity_mps: float = 0.0,
    air_density_kg_m3: float = 1.225,
    duration_s: float = 5.0,
    dt_s: float = 0.001,
) -> TrajectoryData:
    """
    Generate falling object trajectory with drag.

    Physics:
        m*vy'' = m*g - 0.5*rho*Cd*A*vy^2

    Terminal velocity (downward positive):
        vt = sqrt(2*m*g / (rho*Cd*A))

    Solution involves hyperbolic functions. For benchmarking,
    we use explicit Euler integration with small dt.

    Args:
        mass_kg: Object mass
        drag_coef: Drag coefficient Cd (typically 0.1-2.0)
        area_m2: Frontal area
        gravity_mps2: Gravitational acceleration
        initial_height_m: Initial height above ground
        initial_velocity_mps: Initial vertical velocity
        air_density_kg_m3: Air density (default sea level)
        duration_s: Simulation duration
        dt_s: Integration timestep

    Returns:
        TrajectoryData with vertical trajectory
    """
    n_steps = min(int(duration_s / dt_s), 50000)
    dt = duration_s / n_steps

    y = np.zeros(n_steps)  # height
    vy = np.zeros(n_steps)  # velocity (downward positive)
    t = np.zeros(n_steps)

    y[0] = initial_height_m
    vy[0] = initial_velocity_mps

    # Drag constant: k = 0.5 * rho * Cd * A
    k = 0.5 * air_density_kg_m3 * drag_coef * area_m2

    for i in range(1, n_steps):
        # m*dv/dt = m*g - k*v^2
        # dv/dt = g - (k/m)*v^2
        if vy[i-1] >= 0:
            drag_force = k * vy[i-1]**2
        else:
            drag_force = -k * vy[i-1]**2  # Drag always opposes motion

        dvy = gravity_mps2 - (drag_force / mass_kg)
        vy[i] = vy[i-1] + dvy * dt
        y[i] = y[i-1] + vy[i] * dt

        # Stop at ground
        if y[i] < 0:
            y[i] = 0
            vy[i] = 0
            t[i:] = t[i-1] + np.arange(n_steps - i) * dt
            y[i:] = 0
            vy[i:] = 0
            break

        t[i] = t[i-1] + dt

    # Position: x=0, y=height, z=0
    x = np.zeros_like(y)
    z = np.zeros_like(y)
    positions_3d = np.stack([x, y, z], axis=1)

    # Terminal velocity for reference
    terminal_v = np.sqrt(2 * mass_kg * gravity_mps2 / (air_density_kg_m3 * drag_coef * area_m2))

    return TrajectoryData(
        time_s=t,
        positions_3d=positions_3d,
        angles=np.zeros((n_steps, 1)),  # No angles for falling object
        angular_velocities=np.zeros((n_steps, 1)),
        metadata={
            "terminal_velocity_mps": terminal_v,
            "drag_constant_k": k,
            "mass_kg": mass_kg,
            "drag_coef": drag_coef,
            "area_m2": area_m2,
            "dt_s": dt,
            "n_steps": n_steps,
        }
    )


def generate_spring_mass_trajectory(
    mass_kg: float = 1.0,
    stiffness_n_m: float = 100.0,
    damping_n_s_m: float = 2.0,
    gravity_mps2: float = 0.0,  # Usually horizontal, no gravity
    initial_displacement_m: float = 0.1,
    duration_s: float = 10.0,
    dt_s: float = 0.001,
) -> TrajectoryData:
    """
    Generate damped spring-mass trajectory.

    Physics:
        m*x'' + c*x' + k*x = 0

    Solutions:
        - Underdamped (c < 2*sqrt(k*m)): x(t) = A*exp(-zeta*omega_n*t)*cos(omega_d*t + phi)
        - Critically damped (c = 2*sqrt(k*m)): x(t) = (A + B*t)*exp(-omega_n*t)
        - Overdamped (c > 2*sqrt(k*m)): x(t) = A*exp(lambda1*t) + B*exp(lambda2*t)

    Natural frequency: omega_n = sqrt(k/m)
    Damping ratio: zeta = c / (2*sqrt(k*m))

    Args:
        mass_kg: Oscillator mass
        stiffness_n_m: Spring stiffness k
        damping_n_s_m: Damping coefficient c
        gravity_mps2: Gravity (usually 0 for horizontal spring)
        initial_displacement_m: Initial displacement from equilibrium
        duration_s: Simulation duration
        dt_s: Integration timestep

    Returns:
        TrajectoryData with oscillator trajectory
    """
    n_steps = min(int(duration_s / dt_s), 50000)
    dt = duration_s / n_steps

    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    t = np.zeros(n_steps)

    x[0] = initial_displacement_m

    for i in range(1, n_steps):
        # m*x'' + c*x' + k*x = m*g (if vertical)
        # For horizontal: m*x'' + c*x' + k*x = 0
        dxdt = -damping_n_s_m * v[i-1] - stiffness_n_m * x[i-1] + mass_kg * gravity_mps2
        v[i] = v[i-1] + (dxdt / mass_kg) * dt
        x[i] = x[i-1] + v[i] * dt
        t[i] = t[i-1] + dt

    # Position in 3D (x is displacement along x-axis)
    positions_3d = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

    # Analytical properties
    omega_n = np.sqrt(stiffness_n_m / mass_kg)
    zeta = damping_n_s_m / (2 * np.sqrt(stiffness_n_m * mass_kg))

    if zeta < 1:
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        period_s = 2 * np.pi / omega_d
    else:
        period_s = None  # Not oscillatory

    return TrajectoryData(
        time_s=t,
        positions_3d=positions_3d,
        angles=x.reshape(-1, 1),  # Using displacement as "angle" equivalent
        angular_velocities=v.reshape(-1, 1),
        metadata={
            "omega_n": omega_n,
            "zeta": zeta,
            "period_s": period_s,
            "damping_ratio": zeta,
            "dt_s": dt,
            "n_steps": n_steps,
        }
    )


def generate_belt_gantry_trajectory(
    carriage_mass_kg: float = 2.0,
    payload_mass_kg: float = 1.0,
    belt_speed_mps: float = 0.5,
    friction_coef: float = 0.1,
    belt_length_m: float = 2.0,
    gravity_mps2: float = 9.81,
    duration_s: float = 5.0,
    dt_s: float = 0.001,
) -> TrajectoryData:
    """
    Generate belt-gantry mechanism trajectory.

    Physics:
        (m_carriage + m_payload)*x'' = T - friction*x'

    For constant belt speed, the carriage accelerates until
    belt friction equals driving force.

    Steady state velocity when: T = friction * v

    Args:
        carriage_mass_kg: Mass of carriage
        payload_mass_kg: Mass of payload on carriage
        belt_speed_mps: Linear velocity of belt
        friction_coef: Friction coefficient
        belt_length_m: Total belt length
        gravity_mps2: Gravitational acceleration
        duration_s: Simulation duration
        dt_s: Integration timestep

    Returns:
        TrajectoryData with gantry trajectory
    """
    n_steps = min(int(duration_s / dt_s), 50000)
    dt = duration_s / n_steps

    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    t = np.zeros(n_steps)

    total_mass = carriage_mass_kg + payload_mass_kg
    x[0] = 0.0

    # Belt applies velocity to carriage via friction
    for i in range(1, n_steps):
        # Belt velocity
        v_belt = belt_speed_mps

        # Sticking friction (Coulomb)
        if v[i-1] < v_belt:
            # Accelerate toward belt speed
            accel = friction_coef * gravity_mps2 * (v_belt - v[i-1])
        else:
            accel = -friction_coef * gravity_mps2 * (v[i-1] - v_belt)

        v[i] = v[i-1] + accel * dt
        v[i] = np.clip(v[i], 0, v_belt * 1.1)  # Can't exceed belt speed much

        x[i] = x[i-1] + v[i] * dt

        # Boundary
        if x[i] > belt_length_m:
            x[i] = belt_length_m
            v[i] = 0

        t[i] = t[i-1] + dt

    positions_3d = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

    return TrajectoryData(
        time_s=t,
        positions_3d=positions_3d,
        angles=v.reshape(-1, 1),  # Using velocity as "angle" equivalent
        angular_velocities=np.zeros_like(v.reshape(-1, 1)),
        metadata={
            "carriage_mass_kg": carriage_mass_kg,
            "payload_mass_kg": payload_mass_kg,
            "total_mass_kg": total_mass,
            "belt_speed_mps": belt_speed_mps,
            "friction_coef": friction_coef,
            "belt_length_m": belt_length_m,
            "dt_s": dt,
            "n_steps": n_steps,
        }
    )


# Registry of all benchmark scenarios
SCENARIOS: Dict[str, GroundTruthParams] = {
    "calibrated_pendulum": GroundTruthParams(
        masses_kg=(1.0,),
        lengths_m=(0.5,),
        friction_coef=0.0,
        damping=0.0,
        gravity_mps2=9.81,
        initial_angles=(0.5,),
        initial_velocities=(0.0,),
    ),
    "damped_pendulum": GroundTruthParams(
        masses_kg=(1.5,),
        lengths_m=(0.4,),
        friction_coef=0.0,
        damping=0.05,
        gravity_mps2=9.81,
        initial_angles=(0.8,),
        initial_velocities=(0.0,),
    ),
    "2link_arm": GroundTruthParams(
        masses_kg=(1.0, 0.5),
        lengths_m=(0.3, 0.25),
        friction_coef=0.0,
        damping=0.0,
        gravity_mps2=9.81,
        initial_angles=(0.5, 0.3),
        initial_velocities=(0.0, 0.0),
    ),
    "falling_with_drag": GroundTruthParams(
        masses_kg=(2.0,),
        lengths_m=(),
        friction_coef=0.1,
        damping=0.0,
        gravity_mps2=9.81,
        drag_coef=0.8,
        area_m2=0.02,
    ),
    "spring_oscillator": GroundTruthParams(
        masses_kg=(1.0,),
        lengths_m=(),
        friction_coef=0.0,
        damping=0.5,
        gravity_mps2=0.0,
        stiffness=100.0,
    ),
    "belt_gantry": GroundTruthParams(
        masses_kg=(2.0, 1.0),  # carriage, payload
        lengths_m=(),
        friction_coef=0.1,
        damping=0.0,
        gravity_mps2=9.81,
    ),
}


def get_scenario(scenario_name: str) -> Tuple[GroundTruthParams, TrajectoryData]:
    """
    Get ground truth parameters and generate trajectory for a scenario.

    Args:
        scenario_name: Name of the scenario (key in SCENARIOS)

    Returns:
        Tuple of (GroundTruthParams, TrajectoryData)
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(SCENARIOS.keys())}")

    gt = SCENARIOS[scenario_name]

    if scenario_name == "calibrated_pendulum":
        traj = generate_pendulum_trajectory(
            mass_kg=gt.masses_kg[0],
            rod_length_m=gt.lengths_m[0],
            gravity_mps2=gt.gravity_mps2,
            damping=gt.damping,
            initial_angle_rad=gt.initial_angles[0] if gt.initial_angles else 0.5,
            duration_s=10.0,
        )
    elif scenario_name == "damped_pendulum":
        traj = generate_pendulum_trajectory(
            mass_kg=gt.masses_kg[0],
            rod_length_m=gt.lengths_m[0],
            gravity_mps2=gt.gravity_mps2,
            damping=gt.damping,
            initial_angle_rad=gt.initial_angles[0] if gt.initial_angles else 0.8,
            duration_s=10.0,
        )
    elif scenario_name == "2link_arm":
        traj = generate_2link_arm_trajectory(
            link1_mass_kg=gt.masses_kg[0],
            link2_mass_kg=gt.masses_kg[1],
            link1_length_m=gt.lengths_m[0],
            link2_length_m=gt.lengths_m[1],
            gravity_mps2=gt.gravity_mps2,
            initial_angle1_rad=gt.initial_angles[0] if gt.initial_angles else 0.5,
            initial_angle2_rad=gt.initial_angles[1] if gt.initial_angles else 0.3,
            duration_s=10.0,
        )
    elif scenario_name == "falling_with_drag":
        traj = generate_falling_object_trajectory(
            mass_kg=gt.masses_kg[0],
            drag_coef=gt.drag_coef,
            area_m2=gt.area_m2,
            gravity_mps2=gt.gravity_mps2,
            initial_height_m=10.0,
            duration_s=5.0,
        )
    elif scenario_name == "spring_oscillator":
        traj = generate_spring_mass_trajectory(
            mass_kg=gt.masses_kg[0],
            stiffness_n_m=gt.stiffness,
            damping_n_s_m=gt.damping,
            gravity_mps2=gt.gravity_mps2,
            initial_displacement_m=0.1,
            duration_s=10.0,
        )
    elif scenario_name == "belt_gantry":
        traj = generate_belt_gantry_trajectory(
            carriage_mass_kg=gt.masses_kg[0],
            payload_mass_kg=gt.masses_kg[1],
            friction_coef=gt.friction_coef,
            belt_speed_mps=0.5,
            gravity_mps2=gt.gravity_mps2,
            duration_s=5.0,
        )
    else:
        raise ValueError(f"No generator for scenario: {scenario_name}")

    return gt, traj
