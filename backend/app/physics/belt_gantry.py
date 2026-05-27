"""Belt and gantry physics model - v0 first physics target."""
import uuid
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal


class BeltGantryParams(BaseModel):
    belt_tension_N: float = Field(default=20.0, ge=0.0, le=500.0)
    belt_stiffness_Npm: float = Field(default=10000.0, ge=100.0)
    belt_damping: float = Field(default=0.1, ge=0.0, le=1.0)
    carriage_mass_kg: float = Field(default=1.0, ge=0.01)
    carriage_friction: float = Field(default=0.1, ge=0.0, le=1.0)
    carriage_position_m: float = Field(default=0.0, ge=-0.5, le=0.5)
    pulley_radius_m: float = Field(default=0.01, ge=0.001)
    rail_length_m: float = Field(default=1.0, ge=0.1)
    rail_angle_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    motor_torque_Nm: float = Field(default=0.5, ge=0.0)
    motor_rpm: float = Field(default=3000.0, ge=0.0)
    timestep_s: float = Field(default=0.0001, ge=0.00001, le=0.001)
    gravity_mps2: float = 9.81


class SimulationResult(BaseModel):
    simulation_id: str
    horizon_seconds: float
    timesteps: int
    time_array: list[float]
    position_array: list[float]
    velocity_array: list[float]
    acceleration_array: list[float]
    belt_tension_array: list[float]
    belt_vibration_array: list[float]
    kinetic_energy_array: list[float]
    potential_energy_array: list[float]
    total_energy_array: list[float]
    vibration_freq_Hz: float | None = None
    vibration_amplitude_mm: float | None = None
    trajectory_error_mm: float | None = None
    confidence: float = 0.75
    confidence_basis: Literal["classical_simulator"] = "classical_simulator"
    assumptions: list[str] = Field(default_factory=lambda: [
        "Belt as linear spring-damper",
        "Pulley slip ignored",
        "Coulomb + viscous friction",
        "Motor torque constant",
    ])
    parameter_sources: dict = Field(default_factory=lambda: {
        "belt_tension": "estimated_from_video_motion",
        "carriage_mass": "user_override_or_default",
        "friction": "estimated_from_deceleration",
    })
    baseline: dict | None = None
    change_from_baseline: dict | None = None


def simulate_belt_gantry(
    params: BeltGantryParams,
    horizon_seconds: float = 5.0,
    initial_position: float | None = None,
    baseline_params: BeltGantryParams | None = None,
) -> SimulationResult:
    dt = params.timestep_s
    timesteps = max(int(horizon_seconds / dt), 100)
    timesteps = min(timesteps, 50000)

    x0 = initial_position if initial_position is not None else params.carriage_position_m
    x = np.zeros(timesteps)
    v = np.zeros(timesteps)
    a = np.zeros(timesteps)
    tension = np.zeros(timesteps)
    vibration = np.zeros(timesteps)
    t = np.zeros(timesteps)
    ke = np.zeros(timesteps)
    pe = np.zeros(timesteps)
    te = np.zeros(timesteps)

    theta = np.radians(params.rail_angle_deg)
    g_comp = params.gravity_mps2 * np.sin(theta)
    motor_v = params.pulley_radius_m * params.motor_rpm * 2 * np.pi / 60

    x[0] = x0
    tension[0] = params.belt_tension_N

    k = params.belt_stiffness_Npm
    m = params.carriage_mass_kg
    c_belt = params.belt_damping
    mu = params.carriage_friction

    for i in range(1, timesteps):
        stretch = x[i-1] - x0
        F_belt = max(0, params.belt_tension_N + k * stretch - c_belt * (v[i-1] - motor_v))
        F_gravity = -m * g_comp
        if abs(v[i-1]) > 1e-6:
            F_friction = -mu * m * params.gravity_mps2 * np.cos(theta) * np.sign(v[i-1]) - 0.1 * v[i-1]
        else:
            F_friction = -min(abs(F_belt + F_gravity), mu * m * params.gravity_mps2 * np.cos(theta)) * np.sign(F_belt + F_gravity)
        F_net = F_belt + F_gravity + F_friction
        a[i-1] = F_net / m
        v[i] = v[i-1] + a[i-1] * dt
        x[i] = x[i-1] + v[i] * dt
        x[i] = np.clip(x[i], -params.rail_length_m/2, params.rail_length_m/2)
        tension[i] = max(0, params.belt_tension_N + k * (x[i] - x0) - c_belt * (v[i] - motor_v))
        vibration[i] = abs(np.sin(2 * np.pi * np.sqrt(k/m) * t[i])) * 0.001 * tension[i]
        ke[i] = 0.5 * m * v[i]**2
        pe[i] = 0.5 * k * (x[i] - x0)**2
        te[i] = ke[i] + pe[i]
        t[i] = t[i-1] + dt

    try:
        from scipy.fft import fft, fftfreq
        N = len(v)
        freqs = fftfreq(N, dt)
        fft_v = np.abs(fft(v))
        pos_freqs = freqs[:N//2]
        pos_fft = fft_v[:N//2]
        pos_fft[0] = 0
        vib_freq = pos_freqs[np.argmax(pos_fft)]
        vib_amplitude = float(np.std(x)) * 1000
    except ImportError:
        vib_freq = float(np.sqrt(k/m) / (2 * np.pi))
        vib_amplitude = float(np.std(x)) * 1000

    trajectory_error = float(np.sqrt(np.mean((x - np.mean(x))**2)) * 1000)

    result = SimulationResult(
        simulation_id=str(uuid.uuid4())[:8],
        horizon_seconds=horizon_seconds,
        timesteps=timesteps,
        time_array=t.tolist(),
        position_array=x.tolist(),
        velocity_array=v.tolist(),
        acceleration_array=a.tolist(),
        belt_tension_array=tension.tolist(),
        belt_vibration_array=vibration.tolist(),
        kinetic_energy_array=ke.tolist(),
        potential_energy_array=pe.tolist(),
        total_energy_array=te.tolist(),
        vibration_freq_Hz=round(vib_freq, 2),
        vibration_amplitude_mm=round(vib_amplitude, 3),
        trajectory_error_mm=round(trajectory_error, 3),
        confidence=0.75,
        confidence_basis="classical_simulator",
    )

    if baseline_params is not None:
        baseline = simulate_belt_gantry(baseline_params, horizon_seconds, initial_position)
        result.baseline = {
            "vibration_freq_Hz": baseline.vibration_freq_Hz,
            "vibration_amplitude_mm": baseline.vibration_amplitude_mm,
            "trajectory_error_mm": baseline.trajectory_error_mm,
        }
        result.change_from_baseline = {
            "vibration_freq_change_pct": round(100 * (result.vibration_freq_Hz - baseline.vibration_freq_Hz) / baseline.vibration_freq_Hz, 1) if baseline.vibration_freq_Hz else None,
            "vibration_amplitude_change_pct": round(100 * (result.vibration_amplitude_mm - baseline.vibration_amplitude_mm) / baseline.vibration_amplitude_mm, 1) if baseline.vibration_amplitude_mm else None,
            "tension_change_N": round(float(np.mean(result.belt_tension_array)) - float(np.mean(baseline.belt_tension_array)), 2),
        }

    return result


def build_belt_gantry_params(scene_graph) -> BeltGantryParams:
    params = BeltGantryParams()
    for obj in scene_graph.objects:
        if obj.label in ("carriage", "gantry", "head"):
            if obj.physics.get("mass_kg"):
                params.carriage_mass_kg = obj.physics["mass_kg"]
            if obj.physics.get("friction"):
                params.carriage_friction = obj.physics["friction"]
    return params
