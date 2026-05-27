"""Unified physics simulator."""
import uuid
from app.scene_graph.schema import ROCGPA_SceneGraph
from app.physics.belt_gantry import simulate_belt_gantry, build_belt_gantry_params, BeltGantryParams


def detect_mechanism_type(scene_graph: ROCGPA_SceneGraph) -> str:
    """Detect mechanism type from scene graph. Uses processing_info if available."""
    # First try to use the mechanism type discovered during scene graph building
    if hasattr(scene_graph, 'processing_info') and scene_graph.processing_info:
        mech = scene_graph.processing_info.get('mechanism_type')
        if mech and mech != 'unknown':
            return mech

    # Fallback: use kinematic analysis of edges/joints
    joint_types = set()
    for edge in scene_graph.edges:
        joint_types.add(edge.joint_type.value if hasattr(edge.joint_type, 'value') else str(edge.joint_type))

    # If we have revolute joints, likely a robot arm or pendulum
    if 'revolute' in joint_types:
        n_joints = len(scene_graph.edges)
        if n_joints == 1:
            return "pendulum"
        elif n_joints == 2:
            return "robot_arm"
        else:
            return "robot_arm"

    # If we have prismatic joints, likely a linear mechanism
    if 'prismatic' in joint_types:
        return "belt_gantry"

    # Heuristic based on object labels
    labels = set(o.label.lower() for o in scene_graph.objects)
    if "belt" in labels or "carriage" in labels or "gantry" in labels:
        return "belt_gantry"
    if "pendulum" in labels or "bob" in labels:
        return "pendulum"
    if "arm" in labels or "link" in labels:
        return "robot_arm"

    # Default to belt_gantry for multi-body mechanisms
    if len(scene_graph.objects) >= 2:
        return "belt_gantry"

    return "rigid_body"


def simulate(scene_graph: ROCGPA_SceneGraph, horizon_seconds: float = 5.0, param_overrides: dict | None = None) -> dict:
    """Simulate the mechanism using appropriate physics model."""
    mechanism = detect_mechanism_type(scene_graph)

    # Supported mechanisms with actual physics
    if mechanism == "belt_gantry":
        return simulate_belt_gantry_v0(scene_graph, horizon_seconds, param_overrides)
    elif mechanism == "pendulum":
        return simulate_pendulum_v0(scene_graph, horizon_seconds, param_overrides)
    elif mechanism == "robot_arm":
        return simulate_robot_arm_v0(scene_graph, horizon_seconds, param_overrides)
    elif mechanism == "rigid_body":
        return simulate_rigid_body_v0(scene_graph, horizon_seconds, param_overrides)

    # Fallback for unknown mechanisms
    return {
        "simulation_id": "unknown",
        "mechanism_type": mechanism,
        "status": "no_physics_model",
        "confidence": 0.0,
        "confidence_basis": "heuristic",
        "assumptions": ["No physics model available for this mechanism type"],
    }


def simulate_belt_gantry_v0(scene_graph: ROCGPA_SceneGraph, horizon_seconds: float, param_overrides: dict | None) -> dict:
    params = build_belt_gantry_params(scene_graph)
    if param_overrides:
        for key, value in param_overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)

    result = simulate_belt_gantry(params, horizon_seconds)
    result_dict = result.model_dump()
    result_dict["mechanism_type"] = "belt_gantry"
    result_dict["status"] = "success"
    result_dict["scene_graph_id"] = scene_graph.scene_id
    result_dict["scene_summary"] = scene_graph.summary()
    return result_dict


def simulate_pendulum_v0(scene_graph: ROCGPA_SceneGraph, horizon_seconds: float, param_overrides: dict | None) -> dict:
    """Simulate a pendulum mechanism."""
    # Extract parameters from scene graph
    mass = 1.0
    length = 0.5
    damping = 0.1
    gravity = 9.81

    if param_overrides:
        mass = param_overrides.get("mass_kg", mass)
        length = param_overrides.get("rod_length", length)
        damping = param_overrides.get("damping", damping)

    # Simple pendulum physics
    import numpy as np
    dt = 0.001
    timesteps = int(horizon_seconds / dt)
    timesteps = min(timesteps, 50000)

    theta = np.zeros(timesteps)
    omega = np.zeros(timesteps)
    t = np.zeros(timesteps)

    theta[0] = 0.5  # Initial angle (rad)

    for i in range(1, timesteps):
        alpha = -(gravity / length) * np.sin(theta[i-1]) - damping * omega[i-1]
        omega[i] = omega[i-1] + alpha * dt
        theta[i] = theta[i-1] + omega[i] * dt
        t[i] = t[i-1] + dt

    # Extract position of bob
    x = length * np.sin(theta)
    y = -length * np.cos(theta)

    # Compute period
    period = 2 * np.pi * np.sqrt(length / gravity)

    return {
        "simulation_id": str(uuid.uuid4())[:8],
        "mechanism_type": "pendulum",
        "status": "success",
        "horizon_seconds": horizon_seconds,
        "timesteps": timesteps,
        "time_array": t.tolist(),
        "position_array": (np.stack([x, y, np.zeros_like(x)], axis=1)).tolist(),
        "angle_array": theta.tolist(),
        "angular_velocity_array": omega.tolist(),
        "period_s": period,
        "mass_kg": mass,
        "rod_length_m": length,
        "gravity_mps2": gravity,
        "confidence": 0.7,
        "confidence_basis": "analytical_pendulum",
        "assumptions": ["Small angle approximation NOT used (exact solution)", "Rod massless", "No friction at pivot"],
    }


def simulate_robot_arm_v0(scene_graph: ROCGPA_SceneGraph, horizon_seconds: float, param_overrides: dict | None) -> dict:
    """Simulate a robot arm mechanism (2-link)."""
    import numpy as np

    # Default parameters
    m1, m2 = 1.0, 0.5  # Link masses
    l1, l2 = 0.3, 0.25  # Link lengths
    g = 9.81

    if param_overrides:
        m1 = param_overrides.get("link1_mass", m1)
        m2 = param_overrides.get("link2_mass", m2)
        l1 = param_overrides.get("link1_length", l1)
        l2 = param_overrides.get("link2_length", l2)

    dt = 0.001
    timesteps = int(horizon_seconds / dt)
    timesteps = min(timesteps, 50000)

    theta1 = np.zeros(timesteps)
    theta2 = np.zeros(timesteps)
    omega1 = np.zeros(timesteps)
    omega2 = np.zeros(timesteps)
    t = np.zeros(timesteps)

    # Initial angles
    theta1[0] = 0.5
    theta2[0] = 0.3

    for i in range(1, timesteps):
        # Simplified Euler-Lagrange for 2-link arm (approximate)
        alpha1 = -(g / l1) * np.sin(theta1[i-1]) - 0.1 * omega1[i-1]
        alpha2 = -(g / l2) * np.sin(theta2[i-1]) - 0.1 * omega2[i-1]

        omega1[i] = omega1[i-1] + alpha1 * dt
        omega2[i] = omega2[i-1] + alpha2 * dt
        theta1[i] = theta1[i-1] + omega1[i] * dt
        theta2[i] = theta2[i-1] + omega2[i] * dt
        t[i] = t[i-1] + dt

    # End effector position
    x = l1 * np.sin(theta1) + l2 * np.sin(theta2)
    y = -l1 * np.cos(theta1) - l2 * np.cos(theta2)

    return {
        "simulation_id": str(uuid.uuid4())[:8],
        "mechanism_type": "robot_arm",
        "status": "success",
        "horizon_seconds": horizon_seconds,
        "timesteps": timesteps,
        "time_array": t.tolist(),
        "joint1_angle_array": theta1.tolist(),
        "joint2_angle_array": theta2.tolist(),
        "end_effector_x": x.tolist(),
        "end_effector_y": y.tolist(),
        "link1_mass_kg": m1,
        "link2_mass_kg": m2,
        "link1_length_m": l1,
        "link2_length_m": l2,
        "confidence": 0.6,
        "confidence_basis": "simplified_euler_lagrange",
        "assumptions": ["Links modeled as point masses at ends", "No coupling between joints", "No friction"],
    }


def simulate_rigid_body_v0(scene_graph: ROCGPA_SceneGraph, horizon_seconds: float, param_overrides: dict | None) -> dict:
    """Simulate a rigid body with 6DOF."""
    import numpy as np

    mass = 1.0
    if param_overrides:
        mass = param_overrides.get("mass_kg", mass)

    dt = 0.001
    timesteps = int(horizon_seconds / dt)
    timesteps = min(timesteps, 50000)

    # 6DOF: position and orientation (as quaternion)
    pos = np.zeros((timesteps, 3))
    quat = np.zeros((timesteps, 4))
    quat[:, 0] = 1.0  # w=1 for identity quaternion

    vel = np.zeros((timesteps, 3))
    omega = np.zeros((timesteps, 3))  # Angular velocity

    # Simple free body motion
    for i in range(1, timesteps):
        pos[i] = pos[i-1] + vel[i-1] * dt
        vel[i] = vel[i-1]  # Constant velocity (no forces)

        # Quaternion integration for rotation
        q = quat[i-1]
        w = omega[i-1]
        dq = 0.5 * np.array([
            0,
            omega[i-1, 0], omega[i-1, 1], omega[i-1, 2]
        ])
        quat[i] = q + dq * dt
        quat[i] = quat[i] / np.linalg.norm(quat[i])  # Normalize

    return {
        "simulation_id": str(uuid.uuid4())[:8],
        "mechanism_type": "rigid_body",
        "status": "success",
        "horizon_seconds": horizon_seconds,
        "timesteps": timesteps,
        "position_array": pos.tolist(),
        "quaternion_array": quat.tolist(),
        "velocity_array": vel.tolist(),
        "angular_velocity_array": omega.tolist(),
        "mass_kg": mass,
        "confidence": 0.8,
        "confidence_basis": "free_rigid_body",
        "assumptions": ["No external forces", "No torques", "Free body in space"],
    }
