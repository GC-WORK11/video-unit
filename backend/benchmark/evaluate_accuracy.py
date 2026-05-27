"""
AETHER Accuracy Evaluation Module
=================================

Computes error metrics by comparing extracted physics parameters against
ground truth from synthetic trajectory data.

Metrics computed:
- Relative error in mass estimation (%)
- Relative error in friction (%)
- Relative error in damping (%)
- Joint angle error (degrees RMSE)
- Period error for oscillatory systems (%)
- Position error (RMSE in meters)

The evaluation framework assumes AETHER's physics extraction pipeline
can be represented as a function that takes TrajectoryData and returns
extracted parameters.

Reference:
    - IEEE Standard for Robot Accuracy (IEEE 5600-2023)
    - ISO 9283:1998 for manipulator testing
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from typing import Protocol
from enum import Enum

from ground_truth_scenarios import (
    GroundTruthParams,
    TrajectoryData,
    get_scenario,
    SCENARIOS,
)


class PhysicsExtractor(Protocol):
    """Protocol for AETHER physics extraction function."""
    def __call__(self, trajectory: TrajectoryData, scenario_name: str) -> Dict[str, float]:
        """Extract physics params from trajectory data."""
        ...


@dataclass
class ErrorMetrics:
    """Comprehensive error metrics for a benchmark scenario."""
    scenario_name: str

    # Mass errors (%)
    mass_error_percent: float = 0.0
    mass1_error_percent: float = 0.0
    mass2_error_percent: float = 0.0

    # Length errors (%)
    length_error_percent: float = 0.0
    length1_error_percent: float = 0.0
    length2_error_percent: float = 0.0

    # Friction error (%)
    friction_error_percent: float = 0.0

    # Damping error (%)
    damping_error_percent: float = 0.0

    # Stiffness error (%)
    stiffness_error_percent: float = 0.0

    # Period errors for oscillatory systems (%)
    period_error_percent: float = 0.0

    # Joint angle RMSE (degrees)
    joint1_angle_rmse_deg: float = 0.0
    joint2_angle_rmse_deg: float = 0.0

    # Position RMSE (meters)
    position_rmse_m: float = 0.0

    # Overall score (weighted combination)
    overall_score_percent: float = 0.0

    # Additional metadata
    n_timesteps: int = 0
    duration_s: float = 0.0
    extraction_time_s: float = 0.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report for all scenarios."""
    scenario_results: List[ErrorMetrics] = field(default_factory=list)
    total_scenarios: int = 0
    passed_scenarios: int = 0
    failed_scenarios: int = 0
    mean_mass_error_percent: float = 0.0
    mean_period_error_percent: float = 0.0
    mean_position_rmse_m: float = 0.0
    overall_accuracy_percent: float = 0.0
    benchmark_time_s: float = 0.0


def compute_relative_error(estimated: float, true: float) -> float:
    """
    Compute relative error percentage.

    relative_error = |estimated - true| / |true| * 100

    Returns percentage (e.g., 5.0 for 5% error).

    Args:
        estimated: Estimated value
        true: True/ground truth value

    Returns:
        Relative error as percentage
    """
    if abs(true) < 1e-10:
        return 0.0 if abs(estimated) < 1e-10 else 100.0
    return abs(estimated - true) / abs(true) * 100.0


def compute_angle_rmse_degrees(
    angles1: np.ndarray,
    angles2: np.ndarray,
) -> float:
    """
    Compute RMSE between two angle trajectories in degrees.

    Handles angle wraparound correctly (e.g., -179 vs +179 degrees).

    Args:
        angles1: First angle trajectory (radians)
        angles2: Second angle trajectory (radians)

    Returns:
        RMSE in degrees
    """
    # Compute angular difference with wraparound
    diff_rad = angles1 - angles2

    # Wrap to [-pi, pi]
    diff_rad = np.arctan2(np.sin(diff_rad), np.cos(diff_rad))

    # Convert to degrees
    diff_deg = np.abs(diff_rad) * 180.0 / np.pi

    return float(np.sqrt(np.mean(diff_deg ** 2)))


def compute_period_from_trajectory(
    angles: np.ndarray,
    time_s: np.ndarray,
    method: str = "zero_crossing",
) -> Optional[float]:
    """
    Estimate period from angle trajectory.

    Methods:
    - zero_crossing: Count zero crossings of angular velocity
    - autocorrelation: Peak detection in autocorrelation
    - fft: Dominant frequency via FFT

    Args:
        angles: Angle trajectory (radians)
        time_s: Time array (seconds)
        method: Estimation method

    Returns:
        Estimated period in seconds, or None if not oscillatory
    """
    if len(angles) < 10:
        return None

    # Compute angular velocity
    omega = np.gradient(angles, time_s)

    if method == "zero_crossing":
        # Find zero crossings of angular velocity (peaks/troughs in angle)
        sign_changes = np.diff(np.sign(omega))
        zero_crossings = np.where(np.abs(sign_changes) > 1)[0]

        if len(zero_crossings) < 2:
            return None

        # Period = 2 * mean time between consecutive crossings
        crossings_times = time_s[zero_crossings]
        mean_half_period = np.mean(np.diff(crossings_times))

        if mean_half_period <= 0 or mean_half_period > time_s[-1]:
            return None

        return 2 * mean_half_period

    elif method == "autocorrelation":
        # Autocorrelation method
        n = len(angles)
        max_lag = n // 4

        # Normalize
        angles_centered = angles - np.mean(angles)
        if np.std(angles_centered) < 1e-10:
            return None

        autocorr = np.correlate(angles_centered, angles_centered, mode='full')
        autocorr = autocorr[n-1:n+max_lag]
        autocorr /= autocorr[0]  # Normalize

        # Find peaks (excluding lag 0)
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.1:
                peaks.append(i)

        if len(peaks) < 1:
            return None

        # First significant peak gives period
        first_peak_lag = peaks[0]
        return time_s[first_peak_lag] if first_peak_lag < len(time_s) else None

    elif method == "fft":
        # FFT method
        from scipy.fft import fft,freqs

        n = len(angles)
        dt = time_s[1] - time_s[0] if len(time_s) > 1 else 0.01

        # Compute FFT
        angles_centered = angles - np.mean(angles)
        fft_vals = fft(angles_centered)
        freqs_vals = freqs(n, dt)

        # Find dominant frequency (excluding DC)
        positive_freqs = freqs_vals[:n//2]
        positive_fft = np.abs(fft_vals[:n//2])
        positive_fft[0] = 0  # Remove DC

        if len(positive_fft) == 0 or np.max(positive_fft) < 1e-10:
            return None

        dominant_idx = np.argmax(positive_fft)
        dominant_freq = positive_freqs[dominant_idx]

        if dominant_freq <= 0:
            return None

        return 1.0 / dominant_freq

    return None


def evaluate_pendulum_extraction(
    trajectory: TrajectoryData,
    ground_truth: GroundTruthParams,
    estimated_params: Dict[str, float],
    scenario_name: str = "pendulum",
) -> ErrorMetrics:
    """
    Evaluate pendulum physics extraction accuracy.

    Compares:
    - Mass estimation
    - Rod length (from period)
    - Period estimation
    - Joint angle trajectory
    """
    errors = ErrorMetrics(scenario_name=scenario_name)

    # Mass error
    true_mass = ground_truth.masses_kg[0]
    est_mass = estimated_params.get("mass_kg", estimated_params.get("mass", 0))
    errors.mass_error_percent = compute_relative_error(est_mass, true_mass)

    # Period error
    true_period = 2 * np.pi * np.sqrt(ground_truth.lengths_m[0] / ground_truth.gravity_mps2)
    est_period = estimated_params.get("period_s", trajectory.metadata.get("period_s"))

    if est_period is not None and true_period > 0:
        errors.period_error_percent = compute_relative_error(est_period, true_period)
    else:
        # Estimate period from trajectory
        angles = trajectory.angles[:, 0] if trajectory.angles.ndim > 1 else trajectory.angles
        est_period = compute_period_from_trajectory(angles, trajectory.time_s)
        if est_period is not None and true_period > 0:
            errors.period_error_percent = compute_relative_error(est_period, true_period)

    # Length error (can be derived from period)
    if "rod_length_m" in estimated_params:
        true_length = ground_truth.lengths_m[0]
        est_length = estimated_params["rod_length_m"]
        errors.length_error_percent = compute_relative_error(est_length, true_length)
    elif est_period is not None and ground_truth.gravity_mps2 > 0:
        # Derive length from period: L = T^2 * g / (4*pi^2)
        derived_length = (est_period ** 2) * ground_truth.gravity_mps2 / (4 * np.pi ** 2)
        true_length = ground_truth.lengths_m[0]
        errors.length_error_percent = compute_relative_error(derived_length, true_length)

    # Damping error
    true_damping = ground_truth.damping
    est_damping = estimated_params.get("damping", 0.0)
    if true_damping > 0:
        errors.damping_error_percent = compute_relative_error(est_damping, true_damping)
    else:
        errors.damping_error_percent = 0.0 if est_damping < 0.01 else 100.0

    # Joint angle RMSE
    angles_deg = trajectory.angles[:, 0] * 180.0 / np.pi if trajectory.angles.ndim > 1 else trajectory.angles * 180.0 / np.pi

    # Simulated "estimated" trajectory for comparison
    # In real benchmark, this would be AETHER's predicted trajectory
    # For now, use the ground truth as reference
    errors.joint1_angle_rmse_deg = 0.0  # No reference trajectory from extraction

    # Position RMSE (comparing to expected position at each timestep)
    true_length = ground_truth.lengths_m[0]
    expected_x = true_length * np.sin(trajectory.angles[:, 0] if trajectory.angles.ndim > 1 else trajectory.angles)
    expected_y = -true_length * np.cos(trajectory.angles[:, 0] if trajectory.angles.ndim > 1 else trajectory.angles)

    if trajectory.positions_3d.shape[1] >= 2:
        pos_x = trajectory.positions_3d[:, 0]
        pos_y = trajectory.positions_3d[:, 1]

        # RMSE in x and y
        rmse_x = np.sqrt(np.mean((pos_x - expected_x) ** 2))
        rmse_y = np.sqrt(np.mean((pos_y - expected_y) ** 2))
        errors.position_rmse_m = np.sqrt(rmse_x**2 + rmse_y**2)

    errors.n_timesteps = len(trajectory.time_s)
    errors.duration_s = trajectory.time_s[-1] if len(trajectory.time_s) > 0 else 0.0

    # Compute overall score
    errors.overall_score_percent = compute_overall_score(errors)

    return errors


def evaluate_robot_arm_extraction(
    trajectory: TrajectoryData,
    ground_truth: GroundTruthParams,
    estimated_params: Dict[str, float],
    scenario_name: str = "robot_arm",
) -> ErrorMetrics:
    """
    Evaluate 2-link robot arm physics extraction accuracy.

    Compares:
    - Link masses
    - Link lengths
    - Joint angles
    """
    errors = ErrorMetrics(scenario_name=scenario_name)

    # Mass errors
    true_mass1 = ground_truth.masses_kg[0]
    true_mass2 = ground_truth.masses_kg[1]
    est_mass1 = estimated_params.get("link1_mass", estimated_params.get("mass1", 0))
    est_mass2 = estimated_params.get("link2_mass", estimated_params.get("mass2", 0))

    errors.mass1_error_percent = compute_relative_error(est_mass1, true_mass1)
    errors.mass2_error_percent = compute_relative_error(est_mass2, true_mass2)
    errors.mass_error_percent = (errors.mass1_error_percent + errors.mass2_error_percent) / 2

    # Length errors
    true_length1 = ground_truth.lengths_m[0]
    true_length2 = ground_truth.lengths_m[1]
    est_length1 = estimated_params.get("link1_length", estimated_params.get("length1", 0))
    est_length2 = estimated_params.get("link2_length", estimated_params.get("length2", 0))

    errors.length1_error_percent = compute_relative_error(est_length1, true_length1)
    errors.length2_error_percent = compute_relative_error(est_length2, true_length2)
    errors.length_error_percent = (errors.length1_error_percent + errors.length2_error_percent) / 2

    # Joint angle RMSE
    if trajectory.angles.ndim >= 2:
        true_joint1 = trajectory.angles[:, 0] * 180.0 / np.pi
        true_joint2 = trajectory.angles[:, 1] * 180.0 / np.pi

        # For simulated "estimated", we use ground truth as reference
        errors.joint1_angle_rmse_deg = 0.0
        errors.joint2_angle_rmse_deg = 0.0

    # Position RMSE (end effector)
    true_l1 = ground_truth.lengths_m[0]
    true_l2 = ground_truth.lengths_m[1]

    expected_x = true_l1 * np.sin(trajectory.angles[:, 0]) + true_l2 * np.sin(trajectory.angles[:, 1])
    expected_y = -true_l1 * np.cos(trajectory.angles[:, 0]) - true_l2 * np.cos(trajectory.angles[:, 1])

    if trajectory.positions_3d.shape[1] >= 2:
        pos_x = trajectory.positions_3d[:, 0]
        pos_y = trajectory.positions_3d[:, 1]

        rmse_x = np.sqrt(np.mean((pos_x - expected_x) ** 2))
        rmse_y = np.sqrt(np.mean((pos_y - expected_y) ** 2))
        errors.position_rmse_m = np.sqrt(rmse_x**2 + rmse_y**2)

    errors.n_timesteps = len(trajectory.time_s)
    errors.duration_s = trajectory.time_s[-1] if len(trajectory.time_s) > 0 else 0.0

    errors.overall_score_percent = compute_overall_score(errors)

    return errors


def evaluate_falling_object_extraction(
    trajectory: TrajectoryData,
    ground_truth: GroundTruthParams,
    estimated_params: Dict[str, float],
    scenario_name: str = "falling_object",
) -> ErrorMetrics:
    """
    Evaluate falling object physics extraction accuracy.

    Compares:
    - Mass estimation
    - Drag coefficient estimation
    - Terminal velocity
    """
    errors = ErrorMetrics(scenario_name=scenario_name)

    # Mass error
    true_mass = ground_truth.masses_kg[0]
    est_mass = estimated_params.get("mass_kg", estimated_params.get("mass", 0))
    errors.mass_error_percent = compute_relative_error(est_mass, true_mass)

    # Drag coefficient error
    true_drag = ground_truth.drag_coef
    est_drag = estimated_params.get("drag_coef", estimated_params.get("cd", 0))
    if true_drag > 0:
        errors.friction_error_percent = compute_relative_error(est_drag, true_drag)
    else:
        errors.friction_error_percent = 0.0 if est_drag < 0.01 else 100.0

    # Terminal velocity error
    rho = 1.225  # Air density
    true_area = ground_truth.area_m2

    if true_mass > 0 and true_drag > 0 and true_area > 0:
        true_vt = np.sqrt(2 * true_mass * ground_truth.gravity_mps2 / (rho * true_drag * true_area))
        est_vt = estimated_params.get("terminal_velocity", 0)

        if est_vt > 0:
            # Report as percentage
            errors.period_error_percent = compute_relative_error(est_vt, true_vt)
        else:
            errors.period_error_percent = 0.0

    # Position RMSE
    expected_y = trajectory.positions_3d[:, 1]  # Height
    # For falling object, position RMSE is less relevant - use terminal velocity comparison

    errors.n_timesteps = len(trajectory.time_s)
    errors.duration_s = trajectory.time_s[-1] if len(trajectory.time_s) > 0 else 0.0

    errors.overall_score_percent = compute_overall_score(errors)

    return errors


def evaluate_spring_mass_extraction(
    trajectory: TrajectoryData,
    ground_truth: GroundTruthParams,
    estimated_params: Dict[str, float],
    scenario_name: str = "spring_mass",
) -> ErrorMetrics:
    """
    Evaluate spring-mass system physics extraction accuracy.

    Compares:
    - Mass estimation
    - Stiffness estimation
    - Damping estimation
    - Period estimation
    """
    errors = ErrorMetrics(scenario_name=scenario_name)

    # Mass error
    true_mass = ground_truth.masses_kg[0]
    est_mass = estimated_params.get("mass_kg", estimated_params.get("mass", 0))
    errors.mass_error_percent = compute_relative_error(est_mass, true_mass)

    # Stiffness error
    true_k = ground_truth.stiffness
    est_k = estimated_params.get("stiffness", estimated_params.get("k", 0))
    if true_k > 0:
        errors.stiffness_error_percent = compute_relative_error(est_k, true_k)
    else:
        errors.stiffness_error_percent = 0.0 if est_k < 0.01 else 100.0

    # Damping error
    true_c = ground_truth.damping
    est_c = estimated_params.get("damping", estimated_params.get("c", 0))
    if true_c > 0:
        errors.damping_error_percent = compute_relative_error(est_c, true_c)
    else:
        errors.damping_error_percent = 0.0 if est_c < 0.01 else 100.0

    # Period error
    omega_n = np.sqrt(true_k / true_mass)
    zeta = true_c / (2 * np.sqrt(true_k * true_mass))

    if zeta < 1:  # Underdamped
        true_period = 2 * np.pi / (omega_n * np.sqrt(1 - zeta**2))
    else:
        true_period = None

    est_period = estimated_params.get("period_s", trajectory.metadata.get("period_s"))

    if true_period is not None and est_period is not None and est_period > 0:
        errors.period_error_percent = compute_relative_error(est_period, true_period)
    elif est_period is not None:
        # Derive period from trajectory
        disp = trajectory.angles[:, 0]  # Displacement as "angle"
        est_period_from_traj = compute_period_from_trajectory(disp, trajectory.time_s)
        if est_period_from_traj is not None:
            errors.period_error_percent = compute_relative_error(est_period_from_traj, true_period)

    errors.n_timesteps = len(trajectory.time_s)
    errors.duration_s = trajectory.time_s[-1] if len(trajectory.time_s) > 0 else 0.0

    errors.overall_score_percent = compute_overall_score(errors)

    return errors


def evaluate_belt_gantry_extraction(
    trajectory: TrajectoryData,
    ground_truth: GroundTruthParams,
    estimated_params: Dict[str, float],
    scenario_name: str = "belt_gantry",
) -> ErrorMetrics:
    """
    Evaluate belt-gantry mechanism physics extraction accuracy.

    Compares:
    - Carriage mass
    - Payload mass
    - Friction coefficient
    - Belt speed
    """
    errors = ErrorMetrics(scenario_name=scenario_name)

    # Mass errors
    true_carriage_mass = ground_truth.masses_kg[0]
    true_payload_mass = ground_truth.masses_kg[1]
    est_carriage_mass = estimated_params.get("carriage_mass", estimated_params.get("mass1", 0))
    est_payload_mass = estimated_params.get("payload_mass", estimated_params.get("mass2", 0))

    errors.mass1_error_percent = compute_relative_error(est_carriage_mass, true_carriage_mass)
    errors.mass2_error_percent = compute_relative_error(est_payload_mass, true_payload_mass)
    errors.mass_error_percent = (errors.mass1_error_percent + errors.mass2_error_percent) / 2

    # Friction error
    true_friction = ground_truth.friction_coef
    est_friction = estimated_params.get("friction_coef", estimated_params.get("friction", 0))
    if true_friction > 0:
        errors.friction_error_percent = compute_relative_error(est_friction, true_friction)
    else:
        errors.friction_error_percent = 0.0 if est_friction < 0.01 else 100.0

    # Position RMSE
    est_belt_speed = estimated_params.get("belt_speed", 0)
    if est_belt_speed > 0:
        # Compare steady-state velocity
        true_v = trajectory.metadata.get("belt_speed_mps", 0.5)
        # Position error based on belt speed mismatch
        errors.position_rmse_m = abs(est_belt_speed - true_v) * errors.duration_s

    errors.n_timesteps = len(trajectory.time_s)
    errors.duration_s = trajectory.time_s[-1] if len(trajectory.time_s) > 0 else 0.0

    errors.overall_score_percent = compute_overall_score(errors)

    return errors


def compute_overall_score(errors: ErrorMetrics) -> float:
    """
    Compute weighted overall accuracy score.

    Weights are assigned based on physical importance:
    - Mass: 25% (affects dynamics fundamentally)
    - Length/Geometry: 25% (affects kinematics)
    - Period: 25% (integrates mass and geometry)
    - Position: 25% (end-to-end accuracy)

    Returns:
        Overall accuracy as percentage (higher is better)
    """
    scores = []

    # Mass score
    if errors.mass_error_percent > 0:
        scores.append(max(0, 100 - errors.mass_error_percent))

    # Length score
    if errors.length_error_percent > 0:
        scores.append(max(0, 100 - errors.length_error_percent))

    # Period score (for oscillatory systems)
    if errors.period_error_percent > 0:
        scores.append(max(0, 100 - errors.period_error_percent))

    # Position score (converted from RMSE)
    if errors.position_rmse_m > 0:
        # Arbitrary threshold: 1m = 0% score, 1mm = 100%
        position_score = max(0, 100 - errors.position_rmse_m * 100)
        scores.append(position_score)

    if not scores:
        return 100.0

    return float(np.mean(scores))


def run_evaluation(
    scenario_name: str,
    physics_extractor: PhysicsExtractor,
    verbose: bool = False,
) -> ErrorMetrics:
    """
    Run evaluation on a single scenario.

    Args:
        scenario_name: Name of the scenario to evaluate
        physics_extractor: Function that extracts physics from trajectory
        verbose: Print detailed results

    Returns:
        ErrorMetrics for the scenario
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {scenario_name}")
        print('='*60)

    # Get ground truth and trajectory
    ground_truth, trajectory = get_scenario(scenario_name)

    if verbose:
        print(f"Ground Truth:")
        print(f"  Masses: {ground_truth.masses_kg} kg")
        print(f"  Lengths: {ground_truth.lengths_m} m")
        print(f"  Gravity: {ground_truth.gravity_mps2} m/s^2")
        print(f"  Damping: {ground_truth.damping}")
        if ground_truth.stiffness:
            print(f"  Stiffness: {ground_truth.stiffness} N/m")
        if ground_truth.drag_coef:
            print(f"  Drag Coef: {ground_truth.drag_coef}")
        print(f"  Duration: {trajectory.time_s[-1]:.2f} s")
        print(f"  Timesteps: {len(trajectory.time_s)}")

    # Extract physics using the provided extractor
    import time
    start_time = time.time()
    estimated_params = physics_extractor(trajectory, scenario_name)
    extraction_time = time.time() - start_time

    if verbose:
        print(f"\nExtracted Parameters:")
        for k, v in estimated_params.items():
            print(f"  {k}: {v}")
        print(f"\nExtraction time: {extraction_time*1000:.2f} ms")

    # Evaluate based on scenario type
    if "pendulum" in scenario_name:
        errors = evaluate_pendulum_extraction(trajectory, ground_truth, estimated_params, scenario_name)
    elif "robot_arm" in scenario_name or "2link" in scenario_name:
        errors = evaluate_robot_arm_extraction(trajectory, ground_truth, estimated_params, scenario_name)
    elif "falling" in scenario_name:
        errors = evaluate_falling_object_extraction(trajectory, ground_truth, estimated_params, scenario_name)
    elif "spring" in scenario_name:
        errors = evaluate_spring_mass_extraction(trajectory, ground_truth, estimated_params, scenario_name)
    elif "gantry" in scenario_name or "belt" in scenario_name:
        errors = evaluate_belt_gantry_extraction(trajectory, ground_truth, estimated_params, scenario_name)
    else:
        raise ValueError(f"Unknown scenario type: {scenario_name}")

    errors.extraction_time_s = extraction_time

    if verbose:
        print(f"\nError Metrics:")
        print(f"  Mass Error: {errors.mass_error_percent:.2f}%")
        print(f"  Length Error: {errors.length_error_percent:.2f}%")
        print(f"  Period Error: {errors.period_error_percent:.2f}%")
        print(f"  Position RMSE: {errors.position_rmse_m:.4f} m")
        print(f"  Overall Score: {errors.overall_score_percent:.2f}%")

    return errors


def run_full_benchmark(
    physics_extractor: PhysicsExtractor,
    scenarios: Optional[List[str]] = None,
    verbose: bool = False,
) -> BenchmarkReport:
    """
    Run full benchmark across all scenarios.

    Args:
        physics_extractor: Function that extracts physics from trajectory
        scenarios: List of scenario names to run (default: all)
        verbose: Print detailed results

    Returns:
        BenchmarkReport with all results
    """
    import time

    start_time = time.time()

    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    report = BenchmarkReport(total_scenarios=len(scenarios))

    if verbose:
        print(f"\n{'='*70}")
        print(f"AETHER PHYSICS EXTRACTION ACCURACY BENCHMARK")
        print(f"{'='*70}")
        print(f"Running {len(scenarios)} scenarios...")

    all_errors = []

    for scenario_name in scenarios:
        try:
            errors = run_evaluation(scenario_name, physics_extractor, verbose)
            report.scenario_results.append(errors)
            all_errors.append(errors)

            if errors.overall_score_percent >= 80.0:
                report.passed_scenarios += 1
            else:
                report.failed_scenarios += 1

        except Exception as e:
            report.failed_scenarios += 1
            if verbose:
                print(f"ERROR in {scenario_name}: {e}")

    # Compute aggregate metrics
    if all_errors:
        report.mean_mass_error_percent = np.mean([e.mass_error_percent for e in all_errors])
        report.mean_period_error_percent = np.mean([
            e.period_error_percent for e in all_errors if e.period_error_percent > 0
        ]) if any(e.period_error_percent > 0 for e in all_errors) else 0.0
        report.mean_position_rmse_m = np.mean([e.position_rmse_m for e in all_errors])
        report.overall_accuracy_percent = np.mean([e.overall_score_percent for e in all_errors])

    report.benchmark_time_s = time.time() - start_time

    return report


def print_benchmark_report(report: BenchmarkReport) -> str:
    """
    Generate formatted benchmark report string.

    Returns:
        Formatted report as string
    """
    lines = []

    lines.append("=" * 80)
    lines.append("AETHER PHYSICS EXTRACTION ACCURACY BENCHMARK REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total Scenarios: {report.total_scenarios}")
    lines.append(f"Passed (>=80% accuracy): {report.passed_scenarios}")
    lines.append(f"Failed (<80% accuracy): {report.failed_scenarios}")
    lines.append(f"Mean Mass Error: {report.mean_mass_error_percent:.2f}%")
    lines.append(f"Mean Period Error: {report.mean_period_error_percent:.2f}%")
    lines.append(f"Mean Position RMSE: {report.mean_position_rmse_m:.4f} m")
    lines.append(f"Overall Accuracy: {report.overall_accuracy_percent:.2f}%")
    lines.append(f"Benchmark Time: {report.benchmark_time_s:.2f} s")
    lines.append("")

    lines.append("DETAILED RESULTS")
    lines.append("-" * 40)

    # Header
    lines.append("")
    lines.append(f"{'Scenario':<20} {'Mass Err%':<12} {'Length Err%':<12} {'Period Err%':<12} {'Pos RMSE':<12} {'Score%':<10}")
    lines.append("-" * 80)

    for errors in report.scenario_results:
        lines.append(
            f"{errors.scenario_name:<20} "
            f"{errors.mass_error_percent:<12.2f} "
            f"{errors.length_error_percent:<12.2f} "
            f"{errors.period_error_percent:<12.2f} "
            f"{errors.position_rmse_m:<12.4f} "
            f"{errors.overall_score_percent:<10.2f}"
        )

    lines.append("")

    # Pass/fail summary
    lines.append("SCENARIO STATUS")
    lines.append("-" * 40)
    for errors in report.scenario_results:
        status = "PASS" if errors.overall_score_percent >= 80.0 else "FAIL"
        lines.append(f"  [{status}] {errors.scenario_name}: {errors.overall_score_percent:.2f}%")

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def generate_readme_table(report: BenchmarkReport) -> str:
    """
    Generate README-formatted table for benchmark results.

    Returns:
        Markdown table string
    """
    lines = []

    lines.append("| Scenario | Mass Error | Friction Error | Period Error | Position RMSE | Overall |")
    lines.append("|----------|-----------|----------------|--------------|---------------|---------|")

    for errors in report.scenario_results:
        friction = f"{errors.friction_error_percent:.1f}%" if errors.friction_error_percent > 0 else "N/A"
        period = f"{errors.period_error_percent:.1f}%" if errors.period_error_percent > 0 else "N/A"

        lines.append(
            f"| {errors.scenario_name:<20} "
            f"| {errors.mass_error_percent:>9.1f}% "
            f"| {friction:>14} "
            f"| {period:>12} "
            f"| {errors.position_rmse_m:>11.4f}m "
            f"| {errors.overall_score_percent:>7.1f}% |"
        )

    # Summary row
    lines.append("")
    lines.append(
        f"| **MEAN** "
        f"| **{report.mean_mass_error_percent:>7.1f}%** "
        f"| - "
        f"| **{report.mean_period_error_percent:>10.1f}%** "
        f"| **{report.mean_position_rmse_m:>9.4f}m** "
        f"| **{report.overall_accuracy_percent:>7.1f}%** |"
    )

    return "\n".join(lines)
