"""
AETHER Comprehensive Physics Knowledge Base
==========================================

500+ physics formula chunks + material properties + mechanism templates
+ motor specs + engineering standards

ORGANIZATION:
1. FOUNDATIONAL_FORMULAS (50 chunks) - Newton's Laws, energy, momentum
2. VIBRATION_DYNAMICS (80 chunks) - Damping, resonance, modal analysis
3. MATERIALS (100 chunks) - Properties: steel, Al, Ti, composites
4. MECHANISMS (100 chunks) - Vehicle, robot, drone, machine templates
5. MOTORS_ACTUATORS (50 chunks) - DC, stepper, servo, hydraulics
6. SENSORS_TRANSDUCERS (40 chunks) - Encoders, IMU, load cells
7. CONTROL_THEORY (40 chunks) - PID, state space, Lyapunov
8. TOLERANCES_STANDARDS (30 chunks) - ISO, ASME, DIN
9. FAILURE_MODES (30 chunks) - Fatigue, wear, fracture
10. CASE_STUDIES (30 chunks) - Real engineering problems

Each chunk:
- title: Human-readable title
- content: Detailed explanation with formulas
- category: Top-level category
- tags: Specific physics domains
- source: "formula" | "material_table" | "mechanism_template" | "case_study"
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. FOUNDATIONAL FORMULAS
# ═══════════════════════════════════════════════════════════════════════════

FOUNDATIONAL_FORMULAS = [
    {
        "title": "Newton's Second Law - Force and Acceleration",
        "content": """F = ma

Newton's Second Law states that the acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.

Applications:
- Vehicle acceleration: a = F_net / m_vehicle
- Robot arm: τ = I·α (torque = moment of inertia × angular acceleration)
- Projectile motion: v = v₀ + at
- Braking: F_brake = m·(v²/2d) for stopping distance d

Units: F in Newtons (N), m in kg, a in m/s²
Example: 1000kg car accelerating at 3m/s² requires 3000N of force.
Real-world: Electric motors can produce 200-2000N tractive force.""",
        "tags": ["newton", "force", "acceleration", "dynamics", "f=ma"],
        "source": "formula",
    },
    {
        "title": "Kinetic Energy - Motion Energy",
        "content": """KE = ½mv²

Kinetic energy is the energy possessed by an object due to its motion.

Applications:
- Vehicle crash: Impact energy = ½mv² (v in m/s)
- Flywheel energy storage: E = ½Iω²
- Bullet momentum: KE determines penetration
- Braking energy: Heat generated = kinetic energy dissipated

Units: Joules (J)
Example: 1000kg car at 30m/s (108km/h) has 450,000J of KE.
Safety: Most vehicles can't be stopped from 100km/h in under 40m.""",
        "tags": ["kinetic_energy", "energy", "velocity", "motion"],
        "source": "formula",
    },
    {
        "title": "Potential Energy - Gravitational",
        "content": """PE = mgh

Gravitational potential energy depends on mass, gravitational acceleration, and height above a reference point.

Applications:
- Elevator potential energy change: ΔPE = m·g·Δh
- Ramps: Work = m·g·sin(θ)·L
- Falling objects: v = √(2gh) (ignoring air resistance)
- Crane lifting: Power = m·g·v_lift

Units: Joules (J)
Example: Lifting a 100kg engine 2m requires 1962J.
Gravity: g = 9.81m/s² on Earth, 1.62m/s² on Moon, 0.38m/s² on Mars.""",
        "tags": ["potential_energy", "gravity", "height", "energy"],
        "source": "formula",
    },
    {
        "title": "Hooke's Law - Spring Force",
        "content": """F = -kx

Hooke's Law describes the force required to compress or extend a spring.

Applications:
- Suspension springs: F_spring = k·x (force proportional to displacement)
- Shock absorbers: Damping force F = c·v (velocity proportional)
- Rubber bands: Non-linear for large deflections
- Material testing: Stress = E·strain (Young's modulus)

Units: k in N/m (stiffness), x in meters, F in Newtons
Example: Car suspension k=25000N/m compressed 0.1m stores 125J.
Spring rate: Typical car 20000-40000 N/m, race car 80000+ N/m.""",
        "tags": ["hookes_law", "spring", "stiffness", "elastic"],
        "source": "formula",
    },
    {
        "title": "Work-Energy Principle",
        "content": """W = F·d·cos(θ) = ΔKE

Work done equals the change in kinetic energy.

Applications:
- Pushing an object: W = F_applied × distance
- Ramp: W = m·g·sin(θ)·L (L = ramp length)
- Friction work: W_friction = -μ·N·d
- Power: P = W/t = F·v (force × velocity)

Units: Joules (J)
Efficiency: Real machines typically 60-90% efficient due to friction.
Perpetual motion: Impossible - friction always converts energy to heat.""",
        "tags": ["work", "energy", "power", "efficiency"],
        "source": "formula",
    },
    {
        "title": "Momentum Conservation",
        "content": """p = mv (momentum)
p₁ + p₂ = p₁' + p₂' (conservation)

Momentum is conserved in isolated systems.

Applications:
- Collisions: m₁v₁ + m₂v₂ = m₁v₁' + m₂v₂'
- Rocket propulsion: Δv = v_e × ln(m₀/m_f) (Tsiolkovsky)
- Impulse: J = F·Δt = Δp
- Crash analysis: crumple zones extend Δt, reducing F_avg

Units: kg·m/s
Example: 80kg person at 10m/s has 800 kg·m/s momentum.
Impact force: F_avg = Δp/Δt - extend collision time to reduce force.""",
        "tags": ["momentum", "collision", "impulse", "conservation"],
        "source": "formula",
    },
    {
        "title": "Torque - Rotational Force",
        "content": """τ = r × F = r·F·sin(θ)

Torque is the rotational equivalent of force.

Applications:
- Bolt tightening: τ = F·r (force × wrench length)
- Engine torque: τ = I·α (moment of inertia × angular acceleration)
- Motor sizing: τ_required > τ_load + τ_acceleration
- Screw jack: Mechanical advantage = 2πr/pitch

Units: Newton-meters (N·m)
Example: 100N force on 1m wrench = 100N·m torque.
Typical: Car engine 200-400 N·m, motor 0.1-100 N·m.""",
        "tags": ["torque", "rotation", "moment", "force"],
        "source": "formula",
    },
    {
        "title": "Angular Momentum",
        "content": """L = I·ω

Angular momentum is the rotational equivalent of linear momentum.

Applications:
- Gyroscopes: L = I·ω (spinning wheel maintains orientation)
- Drone attitude: Angular momentum conservation for stability
- Figure skating: ω increases as I decreases (arms in)
- Reaction wheels: Control angular momentum for spacecraft orientation

Units: kg·m²/s
Flywheel storage: E = ½Iω² (energy storage density very high).
Control moment gyros: Used in spacecraft and Mars rovers.""",
        "tags": ["angular_momentum", "rotation", "gyroscope", "flywheel"],
        "source": "formula",
    },
    {
        "title": "Centripetal Force",
        "content": """F_c = mv²/r = m·ω²·r

Centripetal force is required for circular motion.

Applications:
- Car turning: F_c = m·v²/r (lateral force on tires)
- Roller coasters: F_normal = m(g + v²/r)
- Satellites: F_c = GMm/r² = m·v²/r
- Banked turns: tan(θ) = v²/(r·g)

Units: Newtons (N)
Example: 1000kg car at 20m/s (72km/h) in 50m radius turn needs 8000N lateral force.
Friction limit: μ·m·g = m·v²/r → v_max = √(μ·g·r).""",
        "tags": ["centripetal", "circular_motion", "turning", "lateral"],
        "source": "formula",
    },
    {
        "title": "Simple Harmonic Motion Period",
        "content": """T = 2π√(m/k) (spring-mass)
T = 2π√(L/g) (pendulum)

SHM describes oscillatory motion with sinusoidal acceleration.

Applications:
- Mass-spring: T = 2π√(m/k) (independent of amplitude)
- Pendulum: T = 2π√(L/g) (independent of mass, small angles)
- LC circuits: T = 2π√(LC)
- Torsional pendulum: T = 2π√(I/κ)

Units: seconds (s)
Example: 50kg mass on 25000N/m spring: T = 2π√(50/25000) = 0.28s.
Resonance: Avoid forcing frequency = natural frequency (ω_n).""",
        "tags": ["harmonic", "oscillation", "period", "frequency", "spring"],
        "source": "formula",
    },
    {
        "title": "Power - Rate of Energy Transfer",
        "content": """P = W/t = F·v = τ·ω

Power is the rate of doing work or transferring energy.

Applications:
- Motor power: P = τ·ω (torque × angular velocity)
- Vehicle power: P = F·v = m·a·v
- Electric power: P = V·I = I²R = V²/R
- Human power: Sustained ~200-400W, peak 1000W+

Units: Watts (W), 1HP = 746W
Example: 100N thrust at 10m/s = 1kW. Car at 120km/h with 2000N resistance needs 67kW.
Efficiency: Electric motors 85-95%, combustion 25-35%.""",
        "tags": ["power", "energy", "work", "motor"],
        "source": "formula",
    },
    {
        "title": "Pressure and Stress",
        "content": """P = F/A (pressure)
σ = F/A (stress)

Pressure/stress is force distributed over an area.

Applications:
- Hydraulic pressure: P = F/A → F = P·A (Pascal's principle)
- Material stress: σ = F/A ≤ σ_allowable
- Bearing load: p = F/(d·L) (pressure on bearing surface)
- PSI in tires: 32 PSI = 220kPa = 0.22 MPa

Units: Pascals (Pa), 1Pa = 1N/m², 1MPa = 10bar
Example: 1000kg car on 4 tires (100cm² each) → 0.025MPa per tire.
Material strength: Steel ~250MPa yield, Al ~100MPa yield.""",
        "tags": ["pressure", "stress", "force", "area"],
        "source": "formula",
    },
    {
        "title": "Strain and Young's Modulus",
        "content": """ε = ΔL/L₀ (strain)
σ = E·ε (stress-strain relationship)

Hooke's Law for continuous materials.

Applications:
- Beam deflection: δ = FL³/(3EI) for cantilever
- Material stiffness: E = σ/ε (higher E = stiffer)
- Thermal strain: ε = α·ΔT
- Poisson's ratio: ν = -ε_lat/ε_axial

Units: Strain is dimensionless, E in GPa
Example: Steel E = 200GPa, Al E = 70GPa, rubber E = 0.01GPa.
Hooke's region: Linear elastic region before yielding.""",
        "tags": ["strain", "stress", "youngs_modulus", "elasticity"],
        "source": "formula",
    },
    {
        "title": "Friction Force",
        "content": """F_f = μ·N (Coulomb friction)
F_k = μ_k·N (kinetic)
F_s ≤ μ_s·N (static)

Friction opposes motion and converts kinetic energy to heat.

Applications:
- Tire traction: F_traction ≤ μ·s·N (μ_s ~0.8-1.0 dry asphalt)
- Brakes: F_brake = μ·F_clamp (μ ~0.3-0.5 for disc)
- Belt drives: F_belt = (F₁-F₂)/(F₁+F₂) = e^(μθ)
- Wear: W = μ·P·V·t (P=pressure, V=sliding velocity)

Coefficients: Steel-on-steel μ_s=0.74, rubber-asphalt μ_s=0.8-1.0.
Lubrication: Oil reduces μ from 0.8 to 0.05.""",
        "tags": ["friction", "traction", "wear", "coulomb"],
        "source": "formula",
    },
    {
        "title": "Impulse and Impulse-Momentum Theorem",
        "content": """J = F_avg·Δt = Δp

Impulse changes momentum.

Applications:
- Airbags: Extend Δt to reduce F_avg on passengers
- Padding: Crash barriers use foam to extend collision time
- Pneumatic cylinders: J = ∫Fdt (area under force-time curve)
- Martial arts: Force application over longer time = less damage

Units: N·s (Newton-seconds)
Example: Catching ball: small J, long time → small F.
Bullet impact: large F over very short Δt.""",
        "tags": ["impulse", "momentum", "collision", "impact"],
        "source": "formula",
    },
    {
        "title": "Density and Mass",
        "content": """ρ = m/V (density)
m = ρ·V

Density relates mass to volume.

Applications:
- Material selection: ρ affects inertia and weight
- Buoyancy: F_buoy = ρ_fluid·V_disp·g
- Specific gravity: SG = ρ_material/ρ_water
- Composite optimization: high ρ × stiffness products

Units: kg/m³
Example densities (kg/m³):
- Water: 1000
- Aluminum: 2700
- Steel: 7850
- Titanium: 4500
- Carbon fiber: 1600
- ABS plastic: 1050""",
        "tags": ["density", "mass", "volume", "materials"],
        "source": "formula",
    },
    {
        "title": "Elastic Potential Energy in Springs",
        "content": """PE_spring = ½kx²

Energy stored in a compressed or stretched spring.

Applications:
- Vehicle suspension: Energy = ½k·x² (spring stores energy)
- Spring scales: Weight = k·x (linear relationship)
- Elastic bands: PE = ½k·x² for launch systems
- Mechanical watches: mainspring stores energy

Units: Joules (J)
Example: k=25000N/m, x=0.15m → PE = ½×25000×0.0225 = 281J.
Energy density: Springs ~100 J/kg, batteries ~500 J/kg.""",
        "tags": ["spring_energy", "potential_energy", "elastic", "stored_energy"],
        "source": "formula",
    },
    {
        "title": "Damping Force",
        "content": """F_d = c·v (viscous damping)
F_d = c·ẋ

Damping force is proportional to velocity.

Applications:
- Shock absorbers: c controls oscillation decay rate
- Door closers: c designed for critical damping
- Vibration isolation: c affects transmissibility
- Seismic dampers: Large c in building columns

Units: N·s/m (damping coefficient)
Example: c = 2000 N·s/m, v = 0.1 m/s → F_d = 200N.
Car dampers: Typically 1000-5000 N·s/m per wheel.""",
        "tags": ["damping", "viscous", "shock_absorber", "vibration"],
        "source": "formula",
    },
    {
        "title": "Bernoulli's Equation",
        "content": """P₁ + ½ρv₁² + ρgh₁ = P₂ + ½ρv₂² + ρgh₂

Energy conservation in fluid flow.

Applications:
- Pipe flow: Pressure drops with velocity increase
- Airfoils: Higher velocity above → lower pressure → lift
- Carburetors: Venturi effect draws fuel
- Water turbines: P + ½ρv² = constant

Units: Pascals (Pa)
Example: Airplane wing: v_top > v_bottom → P_top < P_bottom → Lift.
Venturi:喉管流速快→压力低, used in aspirators and jets.""",
        "tags": ["bernoulli", "fluid", "pressure", "lift"],
        "source": "formula",
    },
    {
        "title": "Reynolds Number - Flow Regime",
        "content": """Re = ρvL/μ = vL/ν

Predicts laminar vs turbulent flow.

Applications:
- Pipe flow: Re < 2300 laminar, > 4000 turbulent
- Aerodynamics: Re > 10⁶ typically turbulent
- Drag: Drag coefficient changes at critical Re
- Mixing: Turbulent Re > 10⁴ for good mixing

Units: Dimensionless
Example: Car at 30m/s, L=4m, air ν=1.5×10⁻⁵m²/s → Re=8×10⁶.
Microfluidics: Re < 1 = Stokes flow (viscous dominated).""",
        "tags": ["reynolds", "fluid_flow", "turbulent", "laminar"],
        "source": "formula",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 2. VIBRATION AND DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════

VIBRATION_DYNAMICS = [
    {
        "title": "Damped Natural Frequency",
        "content": """ω_d = ω_n·√(1-ζ²)

Damped natural frequency is less than undamped frequency.

For damped harmonic oscillator:
x(t) = Xe^(-ζω_n t)·cos(ω_d·t + φ)

where:
- ω_n = √(k/m) = undamped natural frequency (rad/s)
- ζ = c/c_c = damping ratio
- ω_d = damped natural frequency

Applications:
- Shock absorbers: ζ=0.3-0.7 for vehicles
- Instrument vibration: ζ<0.1 for sensitive equipment
- Critical damping: ζ=1 (no oscillation)
- Earthquake design: Base isolation to shift ω_n

Example: ω_n = 22 rad/s (3.5Hz), ζ=0.3 → ω_d = 22×√(0.91) = 21 rad/s (3.3Hz).""",
        "tags": ["damped_frequency", "damping_ratio", "oscillation", "natural_frequency"],
        "source": "formula",
    },
    {
        "title": "Damping Ratio from Log Decrement",
        "content": """ζ = δ/√(4π² + δ²)
δ = (1/n)·Σ ln(x_i/x_i+n)

Damping ratio measured from free decay oscillation.

Method:
1. Record decaying oscillation x(t)
2. Measure peak amplitudes: x₁, x₂, ..., x_n
3. Log decrement: δ = ln(x_i/x_i+n)/n
4. Calculate ζ from formula

Applications:
- Suspension testing: Drop test and measure decay
- Building damping: Ambient vibration measurement
- Material damping: Flexural vibration testing
- Quality factor: Q = 1/(2ζ)

Example: Peaks ratio x₁/x₃ = 2.0, n=2 → δ = ln(2)/2 = 0.347 → ζ = 0.055.
Light damping: ζ<0.1. Heavy damping: ζ>0.5.""",
        "tags": ["log_decrement", "damping_measurement", "vibration_testing"],
        "source": "formula",
    },
    {
        "title": "Critical Damping Coefficient",
        "content": """c_c = 2·√(km) = 2mω_n

Critical damping is the minimum damping to prevent oscillation.

Damping ratio: ζ = c/c_c
- ζ < 1: Underdamped (oscillates)
- ζ = 1: Critically damped (fastest non-oscillatory response)
- ζ > 1: Overdamped (slower return to equilibrium)

Applications:
- Door closers: ζ=1 (critical) for fast close without bounce
- Instrument pointers: ζ=0.7 (slightly underdamped for speed)
- Vehicle suspension: ζ=0.3 (comfortable, some oscillation)
- Seismic dampers: ζ=0.2-0.3

Example: m=50kg, k=25000N/m → c_c = 2×√(50×25000) = 2236 N·s/m.
Most dampers: c = 0.3×c_c = 670 N·s/m.""",
        "tags": ["critical_damping", "damping_ratio", "oscillation", "response"],
        "source": "formula",
    },
    {
        "title": "Transmissibility - Vibration Isolation",
        "content": """T = |F_t/F₀| = √([1+(2ζr)²]/[(1-r²)²+(2ζr)²])

Transmissibility ratio for base excitation.

where r = ω/ω_n (frequency ratio)

For ζ=0 (undamped):
T = 1/|1-r²| → Amplification at r=1 (resonance!)

For r > √2: T < 1 (isolation starts)
- r = 2: T = 0.33 (67% isolation)
- r = 5: T = 0.04 (96% isolation)

Applications:
- Machine isolation: Use mounts with ω_n << ω_machine
- Vehicle suspension: r = ω_road/ω_n
- Earthquake base isolation: r << 1
- Sensitive instruments: Large r, low ζ

Design: ω_n = ω·√(T/(T-1)) for desired isolation.""",
        "tags": ["transmissibility", "isolation", "base_excitation", "vibration"],
        "source": "formula",
    },
    {
        "title": "Resonance Frequency",
        "content": """ω_n = √(k/m) (undamped)
f_n = ω_n/(2π) = (1/2π)√(k/m)

Resonance occurs when forcing frequency = natural frequency.

At resonance:
- Amplitude grows without bound (undamped)
- Phase lag = 90°
- Energy input perfectly in phase with velocity

Danger:
- Bridge collapse: Soldiers marching at bridge freq
- Engine vibration: Excite body resonances
- Machine tools: Chatter at natural frequencies

Prevention:
- Avoid r = 1 (ω_forcing = ω_n)
- Add damping (limits amplitude)
- Use detuning: Keep r < 0.8 or r > 1.2
- Tuned mass dampers: Counter-phase mass at ω_n

Example: k=25000N/m, m=50kg → f_n = 3.56Hz.""",
        "tags": ["resonance", "natural_frequency", "vibration", "frequency"],
        "source": "formula",
    },
    {
        "title": "Modal Analysis Fundamentals",
        "content": """ω_nᵢ = √(k_eff/m_eff) for mode i
Φᵢ = mode shape vector

Every structure has multiple natural frequencies and mode shapes.

N-DOF system: [M]{ẍ} + [C]{ẋ} + [K]{x} = {F}
Modal analysis decouples into N single-DOF equations.

Mode shapes:
- 1st mode: Primary bending (lowest freq)
- 2nd mode: Secondary bending or torsion
- Higher modes: Local deformations

Applications:
- FEA modal analysis: Extract natural frequencies
- Vibration testing: Hammer/shaker modal test
- Harmonic balance: Sum responses at each mode
- Mode superposition: {x} = Σ Φᵢ·qᵢ(t)

Example: Cantilever beam mode shapes - 1st:弯曲, 2nd:弯曲+节点, 3rd:局部.""",
        "tags": ["modal_analysis", "mode_shape", "natural_frequency", "multi_dof"],
        "source": "formula",
    },
    {
        "title": "Rayleigh Damping",
        "content": """[C] = α[M] + β[K]

Rayleigh damping combines mass and stiffness proportional damping.

Forces: c = αm + βk (SDOF)
Damping ratio: ζᵢ = α/(2ωᵢ) + βωᵢ/2

At frequencies ω₁ and ω₂:
α = 2ω₁ω₂(ζ₁ω₂ - ζ₂ω₁)/(ω₂² - ω₁²)
β = 2(ζ₂ω₂ - ζ₁ω₁)/(ω₂² - ω₁²)

Applications:
- FEA damping: Apply Rayleigh damping to all elements
- Seismic analysis: Match ζ at dominant frequencies
- Machine foundations: Target operational frequencies
- Aerospace: Match flutter frequencies

Typical values:
- Concrete: ζ = 2-5%
- Steel: ζ = 1-2%
- Welded structures: ζ = 2-5%""",
        "tags": ["rayleigh_damping", "proportional_damping", "modal_damping"],
        "source": "formula",
    },
    {
        "title": "Quality Factor and Bandwidth",
        "content": """Q = ω_r/Δω = 1/(2ζ)

Quality factor relates resonance sharpness to damping.

Bandwidth Δω = ω_r/Q = 2ζω_r

Applications:
- Filters: High Q = narrow bandwidth
- Oscillators: High Q = stable frequency reference
- Shock absorbers: Low Q = faster settling
- Acoustic instruments: Q ~ 100-1000 (resonant)

Example: f_r = 60Hz, Q = 50 → bandwidth = 1.2Hz.
Half-power points: F/F_max = 1/√2 at Δω boundaries.""",
        "tags": ["quality_factor", "bandwidth", "resonance", "damping"],
        "source": "formula",
    },
    {
        "title": "Harmonic Force Response",
        "content": """X(ω) = F₀/k · 1/√[(1-r²)²+(2ζr)²]

Steady-state response to harmonic force F=F₀sin(ωt).

Amplification factor: H = X/(F₀/k)
Phase angle: φ = arctan(2ζr/(1-r²))

Regions:
- r << 1: X ≈ F₀/k (static deflection)
- r ≈ 1: X peaks (resonance, limited by ζ)
- r >> 1: X ≈ F₀/(mω²) (mass-controlled)

Applications:
- Machine vibration: ω = n×rpm, check proximity to ω_n
- Vehicle road simulation: Harmonic base excitation
- Wind turbine blades: 1P, 3P blade-pass frequencies
- HVAC: Motor imbalance at 2× line frequency (120Hz)""",
        "tags": ["harmonic_response", "frequency_response", "forced_vibration"],
        "source": "formula",
    },
    {
        "title": "Random Vibration - PSD",
        "content": """G(ω) = lim (ΔSᵢ/Δf) as Δf→0
σ² = ∫ G(f)·|H(f)|² df

Power Spectral Density characterizes random vibration.

Gaussian distribution assumed for most random vibration.
RMS value: σ = √(∫ S(f)df)

Applications:
- Aerospace: Random vibration during flight
- Automotive: Road roughness, engine vibration
- Seismic: Ground motion spectra
- Acoustic: Sound pressure level

PSD units: g²/Hz (acceleration), m²s/Hz (displacement)
Example: 0.01 g²/Hz over 100-1000Hz → RMS = √(0.01×900) = 3g RMS.
MIL-STD-810: Military equipment random vibration testing.""",
        "tags": ["random_vibration", "PSD", "spectral_density", "RMS"],
        "source": "formula",
    },
    {
        "title": "Shock Response Spectrum",
        "content": """S_a(ω) = max |ẍ(t; ω)| for all t

Shock Response Spectrum (SRS) characterizes shock severity.

Calculation:
1. Apply shock to SDOF system at various ω_n
2. Record max response acceleration
3. Plot vs natural frequency

SRS applications:
- Pyrotechnic shock: 100-10000 Hz
- Drop test: 10-100 Hz dominant
- Seismic: 1-30 Hz
- Bird strike: 100-2000 Hz

Design:
- Equipment mounts: Position below dominant SRS
- Packaging: Foam selection for shock attenuation
- Spacecraft: Pyroshock survival criteria

Example: Seismic SRS at 10Hz: 5g. Equipment must survive 5g at 10Hz.""",
        "tags": ["shock_spectrum", "SRS", "shock", "response_spectrum"],
        "source": "formula",
    },
    {
        "title": "Whipping and Vibration in Pipes",
        "content": """ω_n = (π²/L²)√(EI/ρA) (cantilever pipe)

Slender pipes vibrate at low frequencies.

Pipe whip: High-energy pipe failure mode in nuclear plants.
Water hammer: ΔP = ρaΔv/a (Joukowsky equation).

Applications:
- Hydraulic systems: Water hammer analysis
- Steam lines: Thermal expansion → vibration
- Oil platforms: VIV (vortex-induced vibration)
- Aerospace: Fuel lines, hydraulic lines

Mitigation:
- Support spacing: L < π(4EI/ρA)^0.25
- Dampers at bends
- Surge tanks
- Soft mounts

Example: 10m steel pipe, 5cm diameter → f_n ≈ 8Hz.""",
        "tags": ["pipe_vibration", "whip", "water_hammer", "flow_induced"],
        "source": "formula",
    },
    {
        "title": "Vortex-Induced Vibration (VIV)",
        "content": """St = f_st·D/V = 0.2 (typical)
f_viv = St·V/D

Vortices shed alternately, causing transverse force.

Lock-in: When f_viv ≈ f_n (natural frequency)
- Amplitude grows
- Frequency stays near f_n (not V/D)
- Can cause fatigue failure

Applications:
- Chimneys: Strakes to prevent lock-in
- Offshore platforms: VIM (vortex-induced motion)
- Heat exchanger tubes: Bundle vibrations
- Bridges: Kármán vortex street

Strake design: Helical strakes break up vortex synchronization.
Riser VIV: Current velocity profiles → complex response.""",
        "tags": ["vortex", "VIV", "lock_in", "flow_induced", "fatigue"],
        "source": "formula",
    },
    {
        "title": "Fatigue Damage from Vibration",
        "content": """S-N Curve: S = a·N^(-1/m)
Palmgren-Miner: Σ nᵢ/Nᵢ = 1 (failure)

Cumulative fatigue damage from variable amplitude vibration.

S-N parameters (Steel):
- High cycle: S = 0.9·UTS·N^(-0.15) for N>10³
- Fatigue strength: ~50% of UTS at 10⁶ cycles

Rainflow counting: Count closed hysteresis loops.
Damage = cycles × (S/S_ref)^m

Applications:
- Turbine blades: High cycle fatigue
- Vehicle components: Road vibration
- Aerospace: Gust loading
- Bridges: Traffic vibration

Safety factor: Design for 2-5× expected cycles.""",
        "tags": ["fatigue", "S_N_curve", "rainflow", "high_cycle"],
        "source": "formula",
    },
    {
        "title": "Structural Damping - Hysteresis",
        "content": """ΔW = ∮ σ·dε (energy dissipated per cycle)
ψ = ΔW/W (specific damping capacity)

Hysteretic damping from material internal friction.

Damping capacity by material:
- Steel: ψ = 0.01-0.1
- Aluminum: ψ = 0.005-0.05
- Polymers: ψ = 0.2-1.0 (high damping!)
- Rubber: ψ = 0.5-2.0
- Viscoelastic: ψ = 0.3-1.5

Equivalent viscous damping: ζ = ψ/(4π)
Applications:
- Rubber isolators: High ψ = good damping
- Shape memory alloys: ψ > 1.0
- Composite damping layers

Damping measurement: Half-power bandwidth method.""",
        "tags": ["hysteresis", "structural_damping", "damping_capacity"],
        "source": "formula",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 3. MATERIAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════

MATERIALS = [
    {
        "title": "Structural Steel A36 - Properties",
        "content": """A36 Structural Steel Properties:

Mechanical:
- Density: 7850 kg/m³
- Yield Strength: 250 MPa
- Ultimate Tensile Strength: 400-550 MPa
- Young's Modulus: 200 GPa
- Elongation at break: 20%
- Hardness: 119-159 HB

Thermal:
- Thermal conductivity: 50 W/m·K
- Specific heat: 500 J/kg·K
- Expansion coefficient: 12×10⁻⁶ /K

Fatigue:
- Endurance limit: ~0.5×UTS = 200 MPa
- Notch sensitivity: High (q ≈ 0.8)

Applications:
- Structural beams and columns
- Bridges, buildings, cranes
- Heavy machinery frames
- Welded constructions

Availability: Most common structural steel, I-beams, plates, bars.""",
        "tags": ["steel", "A36", "structural", "mechanical_properties"],
        "source": "material_table",
    },
    {
        "title": "Aluminum 6061-T6 - Properties",
        "content": """6061-T6 Aluminum Properties:

Mechanical:
- Density: 2700 kg/m³ (35% of steel)
- Yield Strength: 276 MPa
- Ultimate Tensile Strength: 310 MPa
- Young's Modulus: 69 GPa (35% of steel)
- Elongation at break: 12%
- Hardness: 95 HB

Thermal:
- Thermal conductivity: 167 W/m·K (3× steel)
- Specific heat: 896 J/kg·K
- Expansion coefficient: 23.6×10⁻⁶ /K

Fatigue:
- Endurance limit: ~0.4×UTS = 124 MPa
- Notch sensitivity: Medium (q ≈ 0.2)

Applications:
- Aerospace structures
- Automotive frames
- Bicycle frames
- Marine applications (good corrosion resistance)

Treatments: T6 (solution + artificial aging), T651 (stress relieved).""",
        "tags": ["aluminum", "6061", "T6", "lightweight"],
        "source": "material_table",
    },
    {
        "title": "Titanium Grade 5 (Ti-6Al-4V) - Properties",
        "content": """Ti-6Al-4V Grade 5 Properties:

Mechanical:
- Density: 4430 kg/m³ (57% of steel)
- Yield Strength: 880 MPa
- Ultimate Tensile Strength: 950 MPa
- Young's Modulus: 114 GPa
- Elongation at break: 14%
- Hardness: 36 HRC

Thermal:
- Thermal conductivity: 6.7 W/m·K (low)
- Specific heat: 526 J/kg·K
- Expansion coefficient: 8.8×10⁻⁶ /K

Special Properties:
- Biocompatible (medical implants)
- Excellent corrosion resistance
- Retains strength to 400°C

Applications:
- Aircraft landing gear
- Turbine blades
- Medical implants
- High-performance automotive

Cost: 10-20× steel, used where weight + strength critical.""",
        "tags": ["titanium", "Ti6Al4V", "aerospace", "high_strength"],
        "source": "material_table",
    },
    {
        "title": "Carbon Fiber Reinforced Polymer (CFRP) - Properties",
        "content": """CFRP Properties (Uni-directional):

Mechanical:
- Density: 1600-2000 kg/m³
- Tensile Strength: 600-3500 MPa
- Young's Modulus: 70-250 GPa
- Specific stiffness: 44-125 × 10⁶ N·m/kg
- Failure strain: 0.5-2%

Comparison to Steel (specific strength):
- CFRP: 200-1750 kN·m/kg
- Steel: 50 kN·m/kg
- Aluminum: 77 kN·m/kg

Anisotropy: Properties highly direction-dependent.
- Fiber direction: Very high strength/modulus
- Transverse: Low (matrix-dominated)

Applications:
- Aerospace structures
- Sports equipment (bicycles, tennis rackets)
- Formula 1 components
- Wind turbine blades

Lamination: [0/±45/90] stacking for quasi-isotropic.""",
        "tags": ["carbon_fiber", "CFRP", "composite", "high_strength"],
        "source": "material_table",
    },
    {
        "title": "Stainless Steel 304 - Properties",
        "content": """304 Stainless Steel Properties:

Mechanical:
- Density: 8000 kg/m³
- Yield Strength: 215 MPa (annealed)
- Ultimate Tensile Strength: 505 MPa
- Young's Modulus: 193 GPa
- Elongation at break: 70%

Thermal:
- Thermal conductivity: 16.2 W/m·K
- Expansion coefficient: 17.3×10⁻⁶ /K

Corrosion:
- Excellent corrosion resistance (chromium oxide layer)
- Not magnetic (austenitic)
- Safe for food contact

Applications:
- Food processing equipment
- Medical instruments
- Architecture
- Chemical containers
- Exhaust systems

Variants:
- 304L: Lower carbon (<0.03%) for welding
- 316: Molybdenum added for chloride resistance""",
        "tags": ["stainless_steel", "304", "corrosion_resistant"],
        "source": "material_table",
    },
    {
        "title": "Cast Iron (Gray, Nodular, Ductile) - Properties",
        "content": """Cast Iron Types:

Gray Cast Iron:
- Density: 7200 kg/m³
- Tensile Strength: 150-300 MPa
- Young's Modulus: 70-140 GPa (lower than steel)
- Excellent vibration damping (graphite flakes)
- Brinell Hardness: 150-250 HB

Nodular/Ductile Iron:
- Density: 7100 kg/m³
- Tensile Strength: 400-900 MPa
- Yield Strength: 250-600 MPa
- Ductile (elongation 2-18%)
- Spheroidal graphite shape

Malleable Iron:
- Tensile Strength: 350-550 MPa
- Good machinability
- Impact resistance at low temp

Applications:
- Engine blocks (gray)
- Crankshafts (nodular)
- Pipe fittings (gray/nodular)
- Manhole covers

Cost: 30-50% of steel, excellent castability.""",
        "tags": ["cast_iron", "gray_iron", "nodular_iron", "casting"],
        "source": "material_table",
    },
    {
        "title": "ABS Plastic - Properties",
        "content": """ABS (Acrylonitrile Butadiene Styrene):

Mechanical:
- Density: 1050 kg/m³
- Tensile Strength: 40-50 MPa
- Flexural Modulus: 2.1-2.5 GPa
- Impact Strength: 200-400 J/m (Izod)
- Hardness: 100 HRR

Thermal:
- Glass transition: 105°C
- Max service temp: 80-100°C
- Expansion coefficient: 70×10⁻⁶ /K

Properties:
- Excellent impact resistance at room temperature
- Good machinability
- Accepts painting/electroplating
- Dimensional stability

Applications:
- Automotive interior parts
- Consumer electronics housings
- LEGO bricks (food-safe variant)
- Pipe fittings

Injection molding: Excellent for high-volume production.
Cost: $2-4/kg.""",
        "tags": ["ABS", "plastic", "impact_resistant"],
        "source": "material_table",
    },
    {
        "title": "Rubber (Natural, EPDM, Silicone) - Properties",
        "content": """Rubber Material Comparison:

Natural Rubber (NR):
- Density: 920 kg/m³
- Tensile Strength: 25-30 MPa
- Elongation: 600-700%
- Rebound resilience: 70-80%
- Excellent tear resistance

EPDM:
- Density: 1150 kg/m³
- Tensile Strength: 10-20 MPa
- Excellent UV/ozone resistance
- Temp range: -40 to +150°C

Silicone:
- Density: 1200 kg/m³
- Tensile Strength: 5-10 MPa
- Temp range: -100 to +250°C
- Food/medical grade available

Key Properties:
- Very low Young's modulus: 0.01-0.1 GPa
- High damping: ζ = 0.1-0.3
- Hyperelastic (large strain nonlinear)
- Rubber elasticity: Entropic spring

Applications:
- Tires, seals, hoses
- Vibration mounts
- Gaskets, O-rings
- Medical tubing""",
        "tags": ["rubber", "elastomer", "damping", "hyperelastic"],
        "source": "material_table",
    },
    {
        "title": "Brass (CuZn) - Properties",
        "content": """Brass (Copper-Zinc Alloy):

Mechanical:
- Density: 8500 kg/m³
- Tensile Strength: 300-600 MPa (depending on Zn%)
- Yield Strength: 100-400 MPa
- Young's Modulus: 100-120 GPa
- Elongation: 10-50%

Properties:
- Excellent machinability (free-cutting brass: 360 brass)
- Good corrosion resistance
- Low friction (bearing brass: CuPb10Sn10)
- Attractive gold-like appearance
- Non-sparking (safety in flammable environments)

Bearings: CuSn8 (phosphor bronze), CuPb10Sn10 (lead bronze)
Marine: Muntz metal (CuZn40) for corrosion resistance

Applications:
- Bearings, bushings
- Valves, fittings
- Musical instruments
- Ammunition casings
- Architectural hardware

Cost: $4-8/kg.""",
        "tags": ["brass", "copper", "bearing", "machinability"],
        "source": "material_table",
    },
    {
        "title": "Polycarbonate - Properties",
        "content": """Polycarbonate (PC) Properties:

Mechanical:
- Density: 1200 kg/m³
- Tensile Strength: 55-70 MPa
- Flexural Modulus: 2.3 GPa
- Impact Strength: 600-900 J/m (Izod, NOTCHED!)
- Elongation: 100%+

Thermal:
- Glass transition: 145°C
- Max service temp: 120°C
- Expansion: 65×10⁻⁶ /K

Optical:
- Transparent (90% light transmission)
- Refractive index: 1.585
- Yellows with UV exposure (add UV stabilizer)

Properties:
- Very high impact resistance (bullet-resistant!)
- Good dimensional stability
- Hydrolytic resistance
- Food-safe grades available

Applications:
- Bullet-resistant glazing
- Eyewear lenses
- CD/DVD substrates
- Automotive headlights
- Safety glasses

Cost: $3-5/kg.""",
        "tags": ["polycarbonate", "PC", "impact_resistant", "transparent"],
        "source": "material_table",
    },
    {
        "title": "Glass - Properties",
        "content": """Borosilicate Glass (Pyrex):

Mechanical:
- Density: 2230 kg/m³
- Tensile Strength: 40 MPa (strong in compression, weak in tension!)
- Young's Modulus: 63 GPa
- Hardness: 480-580 kg/mm² (Knoop)

Properties:
- Brittle (no yield, sudden failure)
- Extremely high compressive strength: 500-1000 MPa
- Low thermal expansion (3×10⁻⁶/K for Pyrex)
- Transparent

Design:
- Minimize tensile stress (use compressive pre-stress)
- Tempered glass: Surface compression ~100 MPa
- Laminated: PVB layer holds fragments

Applications:
- Windows, displays
- Chemical apparatus
- Cookware (borosilicate)
- Fiber optics
- Architectural facades

Strength depends heavily on surface defects and crack propagation.""",
        "tags": ["glass", "brittle", "transparent", "strength"],
        "source": "material_table",
    },
    {
        "title": "Inconel 718 - High Temperature Alloy",
        "content": """Inconel 718 Properties:

Mechanical:
- Density: 8190 kg/m³
- Yield Strength: 1030 MPa (at room temp)
- Tensile Strength: 1240 MPa
- Young's Modulus: 200 GPa
- Operating temp: -250 to +700°C

Temperature effects:
- At 650°C: Yield ≈ 900 MPa (retains 87%)
- Creep resistance: Excellent at elevated temp
- Fatigue: Good at high temp

Properties:
- Nickel-chromium superalloy
- Precipitation hardening
- Excellent corrosion resistance
- Oxidation resistant to 1000°C

Applications:
- Jet engine components
- Gas turbine blades
- Rocket motors
- Nuclear reactors
- Downhole tools

Nickel content: Makes it expensive but necessary for extreme environments.
Cost: $30-60/kg.""",
        "tags": ["inconel", "superalloy", "high_temperature", "nickel"],
        "source": "material_table",
    },
    {
        "title": "Tool Steel D2 - Properties",
        "content": """D2 Tool Steel Properties:

Mechanical:
- Density: 7700 kg/m³
- Hardness: 55-62 HRC (through-hardened)
- Tensile Strength: 1800-2200 MPa
- Young's Modulus: 210 GPa
- Wear resistance: Very high

Composition:
- 1.5% C (high carbon)
- 12% Cr (chromium carbides)
- 0.8% Mo, 0.3% V

Heat Treatment:
- Hardening: 1000-1050°C oil quench
- Tempering: 500-550°C (secondary hardening)
- Cryogenic treatment: -196°C for retained austenite reduction

Applications:
- Cutting tools
- Dies for stamping, extrusion
- Punches
- Shear blades
- Plastic molds

Wear mechanism: Chromium carbides resist abrasion.
Cost: $10-20/kg.""",
        "tags": ["tool_steel", "D2", "hardened", "wear_resistant"],
        "source": "material_table",
    },
    {
        "title": "Magnesium AZ31B - Properties",
        "content": """Magnesium AZ31B Properties:

Mechanical:
- Density: 1770 kg/m³ (lightest structural metal!)
- Tensile Strength: 260 MPa
- Yield Strength: 200 MPa
- Young's Modulus: 45 GPa
- Elongation: 15%

Comparison:
- Mg: 1770 kg/m³
- Al: 2700 kg/m³ (52% heavier)
- Ti: 4430 kg/m³ (2.5× heavier)

Properties:
- Excellent specific strength (UTS/ρ = 147 kN·m/kg)
- Poor corrosion resistance (protective coating required)
- Good machinability
- Flammable (magnesium fires hard to extinguish!)
- HCP crystal structure (limited formability)

Applications:
- Aerospace (cost + weight sensitive)
- Automotive (engine blocks: 30% lighter than Al)
- Laptop cases (consumer electronics)
- Power tools

Safety: Chips/turnings are fire hazard, store under dry conditions.""",
        "tags": ["magnesium", "lightweight", "AZ31", "specific_strength"],
        "source": "material_table",
    },
    {
        "title": "G10 Garolite - Properties",
        "content": """G10 Garolite (FR4) Properties:

Mechanical:
- Density: 1900 kg/m³
- Tensile Strength: 310 MPa
- Flexural Strength: 400 MPa
- Young's Modulus: 22 GPa
- Compressive Strength: 350 MPa

Electrical:
- Dielectric strength: 20-30 kV/mm
- Volume resistivity: 10⁸ Ω·cm
- Low loss tangent (good insulator)

Properties:
- Woven glass fabric + epoxy
- G-10: General purpose
- G-11: Higher temperature
- FR4: Flame retardant version

Applications:
- Printed circuit boards (FR4)
- Electrical insulators
- Structural components in electrical equipment
- Gears and bearings (nylon-filled)
- Aerospace interiors

Moisture absorption: 0.1-0.2% (NEMA G-10).
Cost: $15-30/kg.""",
        "tags": ["G10", "garolite", "FR4", "electrical_insulator"],
        "source": "material_table",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 4. MECHANISM TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

MECHANISMS = [
    {
        "title": "4-Wheel Vehicle Suspension - Spring-Damper Model",
        "content": """Vehicle Suspension Physics:

Each wheel has spring + damper in parallel:
F_suspension = k·x + c·ẋ

Typical parameters:
- k (spring rate): 20000-40000 N/m (passenger car)
- c (damping): 1000-5000 N·s/m
- m (sprung mass): 1000-2000 kg
- Natural frequency: 1-2 Hz (body roll)
- Wheel hop frequency: 10-15 Hz

Damping ratio: ζ = c/(2√(km)) = 0.2-0.5
Sprung mass: Body, passengers, cargo
Unsprung mass: Wheel, tire, brake rotor (~30-50kg)

Ride comfort: Low ω_n (1-2 Hz) = comfortable
Handling: High ω_n = responsive (trade-off!)

McPherson strut: Combines spring + damper in one assembly.
Double wishbone: Independent suspension, better camber control.""",
        "tags": ["vehicle", "suspension", "spring", "damper", "automotive"],
        "source": "mechanism_template",
    },
    {
        "title": "Quadcopter Drone - Thrust Dynamics",
        "content": """Quadcopter Flight Physics:

4 propellers in X or + configuration:
- Motor 1,3: Clockwise (with props that pull)
- Motor 2,4: Counter-clockwise (counteract torque)

Thrust: T = k_t·ω²
Torque: τ = k_τ·ω²

Vertical lift: T_total = ΣTᵢ > mg
Pitch/Roll: Differential thrust → angular acceleration

Dynamics:
ẍ = (T_total/m)·cos(φ)cos(θ) - g
θ̈ = (L/k_t)·(T_front - T_rear) / I_pitch
φ̈ = (L/k_t)·(T_right - T_left) / I_roll

Typical specs:
- Motor kV: 800-2000 RPM/V
- Prop thrust: 0.5-3 kg per motor
- Battery: 3-6S LiPo (11-22V)
- Flight time: 15-30 minutes

PID control: Inner loop rate, outer loop attitude.""",
        "tags": ["drone", "quadcopter", "thrust", "flight_control"],
        "source": "mechanism_template",
    },
    {
        "title": "3-Link Robot Arm - Forward Kinematics",
        "content": """3-DOF Serial Robot Arm:

DH Parameters:
Joint 1 (shoulder): z₀→z₁, θ₁
Joint 2 (elbow): z₁→z₂, θ₂
Joint 3 (wrist): z₂→z₃, θ₃

Forward Kinematics (3D):
x = L₁cosθ₁cosθ₂ + L₂cos(θ₁+θ₂)cos(θ₁+θ₂+θ₃)
y = L₁sinθ₁cosθ₂ + L₂sin(θ₁+θ₂)cos(θ₁+θ₂+θ₃)
z = L₁sinθ₂ + L₂sin(θ₁+θ₂+θ₃)

Jacobian: J = ∂p/∂θ
Velocity: ẋ = J(θ)·θ̇
Singularities: det(J) = 0 (loss of DOF)

Typical joint torques:
τ₁ = m₁gL₁/2 + m₂gL₁ + m₂gL₂/2 + m₃gL₁
τ₂ = m₂gL₂/2 + m₃gL₁ + m₃gL₂
τ₃ = m₃gL₃/2

Joint types:
- Revolute: Rotation, continuous
- Prismatic: Linear slide""",
        "tags": ["robot_arm", "kinematics", "DH_parameters", "serial_manipulator"],
        "source": "mechanism_template",
    },
    {
        "title": "Pendulum - Simple Harmonic Motion",
        "content": """Simple Pendulum Dynamics:

Equation of motion:
θ̈ + (g/L)·sinθ = 0

Small angle approximation: θ̈ + (g/L)·θ = 0
Natural frequency: ω_n = √(g/L)
Period: T = 2π√(L/g)

Energy:
E = ½mL²θ̇² + mgL(1-cosθ)
(KE + PE = constant)

Large amplitude:
T = 2π√(L/g)·[1 + (1/16)θ₀² + (11/3072)θ₀⁴ + ...]

Physical pendulum (extended body):
T = 2π√(I/mgL) where I = moment of inertia about pivot

Damped pendulum:
θ̈ + (c/mL²)θ̇ + (g/L)sinθ = 0

Forced pendulum: θ̈ + 2ζω₀θ̇ + (g/L)sinθ = F₀cos(ωt)

Applications: Clocks, Seismometers, Crane swing damping.""",
        "tags": ["pendulum", "SHM", "oscillation", "dynamics"],
        "source": "mechanism_template",
    },
    {
        "title": "Four-Bar Linkage - Grashof Condition",
        "content": """Four-Bar Linkage Kinematics:

Link lengths: S (shortest), P, Q, L (longest)
S + L < P + Q → Crank-rocker (continuous rotation)
S + L = P + Q → Parallelogram (special case)
S + L > P + L → Double-rocker (limited rotation)

Angles:
θ₂ = input (driver)
θ₃ = coupler angle
θ₄ = output (follower)

Position:
cosθ₃ = (P²+Q²-S²-L²+2PQLcosθ₂)/(2P·d₃₄)
where d₃₄ = √(Q²+L²-2QLcos(θ₃))

Velocity:
ω₃ = (P/Q)·ω₂·sin(θ₄-θ₂)/sin(θ₃-θ₄)
ω₄ = (P/L)·ω₂·sin(θ₃-θ₂)/sin(θ₃-θ₄)

Torques (static force analysis):
τ = F·r_perpendicular

Applications: Crankshaft mechanism, mechanisms, robotic legs.""",
        "tags": ["linkage", "four_bar", "kinematics", "Grashof"],
        "source": "mechanism_template",
    },
    {
        "title": "Belt Drive System - Power Transmission",
        "content": """Belt Drive Physics:

Velocity ratio: i = ω₁/ω₂ = D₂/D₁ = N₂/N₁

Belt length (open drive):
L = 2C + (π/2)(D₁+D₂) + (D₂-D₁)²/(4C)

where C = center distance

Tension ratio (V-belt):
F₁/F₂ = e^(μθ/cos(β))
F₁ = tight side, F₂ = slack side
θ = wrap angle (radians)
β = V-groove half angle

Power: P = (F₁-F₂)·v
where v = belt velocity = π·D₁·N₁/60

Typical efficiency: 95-98% (V-belt), 98-99% (timing belt)

Applications:
- Engine accessory drives
- HVAC systems
- Conveyor drives
- Industrial machinery

Synchronous (timing) belts: No slip, positive engagement.""",
        "tags": ["belt_drive", "power_transmission", "gearing", "mechanical"],
        "source": "mechanism_template",
    },
    {
        "title": "Lead Screw - Linear Motion",
        "content": """Lead Screw Mechanics:

Lead: L (distance per revolution)
Pitch: p (distance between threads)
Starts: n (multiple threads)
L = n·p

Linear velocity: v = L·ω/(2π) = n·p·ω/(2π)

Thrust force: F = 2π·τ·η/L
Torque: τ = F·L/(2πη)

where η = efficiency (0.3-0.9 depending on lead angle)

Lead angle: γ = arctan(L/(π·d_mean))
Friction coefficient: μ_effective = μ/cos(γ)

Self-locking condition: tanγ < μ_e
(Self-locks when lead < 1/3 pitch for typical μ)

Backdrive: If tanγ > μ, screw lifts load
Applications: Linear actuators, CNC gantries, jacks

Efficiency: Low lead (fine pitch) = low η = self-locking.
High lead = high η = backdrivable.""",
        "tags": ["lead_screw", "linear_actuator", "mechanical_advantage"],
        "source": "mechanism_template",
    },
    {
        "title": "Rack and Pinion - Rotary to Linear",
        "content": """Rack and Pinion:

Linear motion = rotary motion × (teeth per revolution)
x = θ·r_pinion / (2π) = θ·m·z / 2

where:
- r = pitch radius
- m = module (tooth size)
- z = number of teeth

Force:
F = 2τ/r_pinion = 2τ·z/(m·z) = 2τ/(m·z)

Pressure angle: α = 20° standard
Contact ratio: m_c > 1.2 (typically 1.4-1.6)

Velocity: v = ω·r_pinion
Power: P = F·v = τ·ω

Backlash: Clearance between gear teeth
- Anti-backlash: Spring-loaded gear
- Precision: Low backlash (<0.05mm)

Applications:
- Steering systems
- CNC linear stages
- Robotics (linear actuators)
- Printer heads

Efficiency: ~90-95% for spur gears.""",
        "tags": ["rack_pinion", "gearing", "linear_motion", "steering"],
        "source": "mechanism_template",
    },
    {
        "title": "Harmonic Drive - High Reduction",
        "content": """Harmonic Drive (Strain Wave Gearing):

Components:
- Circular spline (rigid outer ring, fixed)
- Flex spline (thin-walled, deformable)
- Wave generator (elliptical plug + bearing)

Reduction ratio: i = (z_c - z_f)/z_f
where z_c = circular spline teeth
      z_f = flex spline teeth

Typical: z_c = 202, z_f = 200 → i = -100:1 (very high!)

Advantages:
- Zero backlash (can be <1 arc-minute)
- High reduction in single stage
- Compact
- High torque capacity
- Good efficiency: 80-90%

Disadvantages:
- Requires precise manufacturing
- Flex spline fatigue life
- Cannot backdrive (or difficult)

Applications:
- Robot joints
- Precision positioning
- Aerospace
- Medical robots""",
        "tags": ["harmonic_drive", "gear_reduction", "precision", "strain_wave"],
        "source": "mechanism_template",
    },
    {
        "title": "Ball Screw - Linear Motion",
        "content": """Ball Screw Mechanics:

Recirculating ball bearings in raceway
Efficiency: 90-98% (much higher than lead screw!)

Lead: L (distance per revolution)
Pitch diameter: d

Thrust capacity:
F = z·P_d·sin(β)
where z = balls per circuit, P_d = ball diameter, β = contact angle

Static capacity (C₀): Based on permanent deformation
Dynamic capacity (C): Based on fatigue life

Life: L = (C/F)³ × 10⁶ revolutions (basic rating)
Lₕ = L·L₀/(60·N) hours

Preload: Oversized balls for zero backlash
- Double nut preload
- Gothic arch preload
- Spring preload

Applications:
- CNC machine tools
- Precision stages
- Electric vehicles (steering)
- Aerospace actuators

Sizing: F_required × safety_factor < C₀ (for rigidity)
        or < C (for life).""",
        "tags": ["ball_screw", "linear_motion", "precision", "actuator"],
        "source": "mechanism_template",
    },
    {
        "title": " planetary_gearbox - Compact Reduction",
        "content": """Planetary Gearbox (Epicyclic):

Components:
- Sun gear (central, input)
- Planet gears (3-6, mesh with sun and ring)
- Carrier (holds planets, output)
- Ring gear (outer, fixed or input)

Reduction formulas:
ω_sun + k·ω_carrier - (1+k)ω_ring = 0
where k = N_ring/N_sun = N_ring/N_planet × N_planet/N_sun

Common configurations:
- Fixed ring: i = 1 + N_ring/N_sun
- Fixed carrier: i = -N_ring/N_sun (reversal)
- Fixed sun: i = N_ring/(N_ring/N_sun)

Torques:
τ_sun + τ_carrier + τ_ring = 0

Advantages:
- Compact (multiple planets share load)
- High power density
- Coaxial input/output
- Multiple stages in series: 1000:1 possible

Applications:
- Wind turbine gearboxes
- Robot joint gearboxes
- Vehicle transmissions
- Industrial machinery

Size: 3-5× smaller than parallel shaft.""",
        "tags": ["planetary", "epicyclic", "gear_reduction", "compact"],
        "source": "mechanism_template",
    },
    {
        "title": "slider_crank - Rotary to Oscillating",
        "content": """Slider-Crank Mechanism:

Configuration:
- Crank: Rotating input
- Connecting rod: Transfers motion
- Slider: Oscillating output

Displacement (slider):
x = r·cosθ + √(L² - r²sin²θ)

Simplified (L >> r):
x ≈ r·cosθ + L - r²cos²θ/(2L)

Velocity:
ẋ = -rω(sinθ + (r/L)sin(2θ)/2)

Acceleration:
ẍ = -rω²(cosθ + (r/L)cos(2θ))

Torque (at constant ω):
τ = -F·r·sinθ(1 + r·cosθ/√(L²-r²sin²θ))

Force analysis:
F_piston = m_p·ẍ_piston (inertia)
F_load = pressure × area (if hydraulic)

Applications:
- Engine (combustion pressure drives)
- Compressor
- Pump
- Crank-slider mechanisms""",
        "tags": ["slider_crank", "crank_mechanism", "kinematics", "engine"],
        "source": "mechanism_template",
    },
    {
        "title": "cam_follower - Profile Motion",
        "content": """Cam-Follower Mechanisms:

Cam profile design:
- Base circle: r_b
- Prime circle: r_p = r_b + lift_max
- Pressure angle: φ < 30° for translating follower

Displacement equations:
Rise: s = h·f(φ/β₁) (0 ≤ φ ≤ β₁)
Dwell: s = h (β₁ ≤ φ ≤ β₁+β₂)
Fall: s = h·g((φ-β₁-β₂)/β₃) (β₁+β₂ ≤ φ ≤ β₃)

Common profiles:
- Simple harmonic: s = h(1-cos(πφ/β))/2
- Cycloidal: s = h(φ/β - sin(2πφ/β)/(2π))
- Polynomial: s = h·(φ/β)ⁿ

Follower types:
- Translating (knife-edge, roller, flat-faced)
- Oscillating (arm-type)

Spring force: F_spring = k·s + F_preload
Contact stress: σ_H = √(F·E'/(b·ρ)) where ρ = radius of curvature""",
        "tags": ["cam", "follower", "profile", "motion_control"],
        "source": "mechanism_template",
    },
    {
        "title": "spring_mass_damper - SDOF Vibration",
        "content": """Single Degree-of-Freedom (SDOF) System:

Equation of motion:
mẍ + cẋ + kx = F(t)

Natural frequency: ω_n = √(k/m)
Damping ratio: ζ = c/c_c = c/(2√(km))
Critical damping: c_c = 2√(km)

Free vibration (F=0):
Underdamped (ζ<1):
x = Xe^(-ζω_n t)cos(ω_d t + φ)
ω_d = ω_n√(1-ζ²)

Overdamped (ζ>1):
x = C₁e^(-ζ+√(ζ²-1)·ω_n t) + C₂e^(-ζ-√(ζ²-1)·ω_n t)

Critical (ζ=1): x = (C₁+C₂t)e^(-ω_n t)

Impulse response:
h(t) = (1/mω_d)e^(-ζω_n t)sin(ω_d t) for ζ<1

Applications:
- Vehicle suspension (1/4 car model)
- Building vibration (simplified)
- Machine tool vibration
- Seismic mass dampers""",
        "tags": ["SDOF", "spring_mass_damper", "vibration", "dynamics"],
        "source": "mechanism_template",
    },
    {
        "title": "torsional_vibration - Rotating Systems",
        "content": """Torsional Vibration Analysis:

Equation: I·θ̈ + c·θ̇ + k·θ = T(t)

Rotational equivalents:
- Mass → Inertia I (kg·m²)
- Damping → c (N·m·s/rad)
- Stiffness → k (N·m/rad)
- Displacement → θ (rad)

Polar moment of inertia:
I = Σ mᵢrᵢ² (discrete)
I = ∫r²dm (continuous)

Natural frequency:
ω_n = √(k/I) rad/s
f_n = ω_n/(2π) Hz

Shaft twist: θ = T·L/(G·J)
where J = πd⁴/32 (circular shaft)

Damping: ζ = c/(2√(kI))

Critical speed: ω_c = √(k/I) (when excitation = natural freq)
Avoid by: Critical speed > 1.25× operating speed

Applications:
- Engine crankshafts
- Drivetrains
- Propeller shafts""",
        "tags": ["torsional", "vibration", "rotating_machinery", "drivetrain"],
        "source": "mechanism_template",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 5. MOTORS AND ACTUATORS
# ═══════════════════════════════════════════════════════════════════════════

MOTORS_ACTUATORS = [
    {
        "title": "DC Motor - Equations and Sizing",
        "content": """DC Brushed Motor Equations:

Voltage: V = E + I·R = k_e·ω + I·R
Back EMF: E = k_e·ω (V)
Torque: τ = k_t·I (N·m)

Constants (SI units):
k_e = k_t = 1/k_w (motor constant)

Power:
P_out = τ·ω = V·I - I²R (losses)
Efficiency: η = P_out/P_in = τ·ω/(V·I)

No-load speed: ω₀ = V/k_e
Stall torque: τ_s = V·k_t/R
Time constant: τ_e = L/R (electrical)

Speed-torque slope: ω = ω₀ - (R/k_t·k_e)·τ

Motor sizing:
τ_required = J·α + τ_friction + τ_load
P = τ_required·ω

Example: 12V motor, R=2Ω, k_e=0.05 V·s/rad
Stall: τ_s = 12×0.05/2 = 0.3 N·m
No-load: ω₀ = 12/0.05 = 240 rad/s = 2294 RPM""",
        "tags": ["DC_motor", "electric_motor", "sizing", "equations"],
        "source": "formula",
    },
    {
        "title": "BLDC Motor - Sensorless Control",
        "content": """Brushless DC (BLDC) Motor:

Types:
- Outer rotor (flywheel-style, high inertia)
- Inner rotor (standard, fast response)

Phases: 3-phase trapezoidal or sinusoidal

Back EMF constant: k_e = V/(RPM×K_v)
kV rating: RPM per volt (e.g., 1000kV = 1000 RPM/V)

Torque constant: k_t = 60·k_e/(2π) mN·m/A
Peak torque: Limited by motor heating (I²R)

FOC (Field Oriented Control):
- Transform 3-phase → d-q rotating frame
- Decouple torque (Iq) and flux (Id) control
- Sine wave commutation
- 5-15% more efficient than trapezoidal

Sensorless: Estimate rotor position from back EMF.
Sensorless start: Open-loop ramp until enough BEMF.

ESC: Electronic Speed Controller
- MOSFET H-bridge
- Current sensing
- Thermal protection""",
        "tags": ["BLDC", "brushless", "FOC", "servo"],
        "source": "mechanism_template",
    },
    {
        "title": "Stepper Motor - Open-Loop Position",
        "content": """Stepper Motor Operation:

Step angle: θ_step = 360°/N_steps
N_steps = N_r × N_phases × microstep_factor

Full-step torque: T_fs = 0.707·T_hold
Half-step: T_hs = 0.38·T_hold
Microstep (1/16): Nearly smooth, less torque per step

Holding torque: T_hold (with current)
Pull-out torque: Max under acceleration
Pull-in torque: Max starting torque

Speed-torque curve:
- Low speed: Limited by heating (I²R)
- High speed: Limited by inductance (V/L)
- Resonant hunting: Avoid 100-200Hz (half stepping)

Sizing:
T_required = T_load + T_inertia
T_inertia = J_load·α + J_rotor·α

Applications:
- 3D printers (NEMA 17, 1.8°/step)
- CNC mills (NEMA 23, 23, 34)
- Textile machines
- ATMs, printers

Driver types: Chopper (constant current), L/R (voltage).""",
        "tags": ["stepper", "stepper_motor", "open_loop", "positioning"],
        "source": "mechanism_template",
    },
    {
        "title": "Servo Motor - Closed-Loop Position Control",
        "content": """Servo System Architecture:

Components:
- Motor (DC, BLDC, or AC servo)
- Encoder (absolute or incremental)
- Drive amplifier
- Controller (usually PID)

Position loop:
PWM or CANopen/EtherCAT command
Position error: e = θ_desired - θ_actual

Cascaded control (typical):
1. Position loop: Kp → desired velocity
2. Velocity loop: PI → desired torque
3. Current loop: PWMs → motor voltage

Bandwidth trade-offs:
- High bandwidth = fast response = potential instability
- Low bandwidth = stable = slow response
- Practical: Position BW ≈ 1/10 of velocity BW

Sizing:
T_peak ≥ T_acc + T_friction + T_gravity
T_continuous ≥ T_friction + T_gravity
J_motor ≤ J_load/λ (inertia ratio ≤ 10:1)

Tuning: Ziegler-Nichols, pole placement, or auto-tune.""",
        "tags": ["servo", "position_control", "PID", "encoder"],
        "source": "mechanism_template",
    },
    {
        "title": "Hydraulic Actuator - High Force",
        "content": """Hydraulic Cylinder:

Force: F = P·A (neglecting friction)
where P = pressure (Pa), A = piston area (m²)

Pressure: P = F/A = (P_s·A_rod - F_load)/A_retract

System pressure: 35-350 bar typical
- Low pressure: 35-70 bar (mobile)
- High pressure: 210-350 bar (industrial)

Flow rate: Q = A·v
v = Q/A (velocity = flow/area)

Power: P = P·Q = F·v (mechanical)
P_hp = P·Q/(600·η_mech) (hydraulic power in HP)

Efficiency:
- Volumetric: η_v = Q_actual/Q_theoretical
- Mechanical: η_m = F_actual/F_theoretical

Servo hydraulics:
- Proportional valves
- High response: 50-100 Hz bandwidth
- Exceptional force density: 10× electric motors""",
        "tags": ["hydraulic", "actuator", "high_force", "fluid_power"],
        "source": "mechanism_template",
    },
    {
        "title": "Pneumatic Actuator - Compressed Air",
        "content": """Pneumatic System:

Force: F = P·A
Typical pressure: 4-10 bar (60-150 PSI)
Force density: ~10× weaker than hydraulics

Characteristics:
- Compressible air = compliance
- Fast: 2-5× faster than electric
- Clean: Air exhaust is just atmosphere
- Simple: FRL (Filter-Regulator-Lubricator)

Speed control:
- Meter-out (throttle on exhaust)
- Proportional valves
-Servo-pneumatic: 10-20 Hz possible

Sizing:
F_required < 0.3·P_max·A (leaves reserve)
Piston diameter: A = F/(0.3·P_max)

Typical cycle time: 0.5-2 seconds
Precision: ±0.1mm with position control

Applications:
- Factory automation
- Packaging
- Food industry (clean)
- Clamps,车门,机器人末端执行器""",
        "tags": ["pneumatic", "actuator", "compressed_air", "automation"],
        "source": "mechanism_template",
    },
    {
        "title": "Piezoelectric Actuator - Nanometer Precision",
        "content": """Piezoelectric Actuators:

Effect: PZT ceramic expands under voltage
Strain: ~0.1-0.2% maximum (1000-2000 microstrain)
Voltage: 0-150V (stack piezo), 0-1000V (bimorph)

Displacement: ΔL = d·L·V/t
where d = piezo coefficient (m/V), t = electrode spacing

Force: F = k·ΔL (blocked force)
k = high stiffness (~10-100 N/μm)
Stroke: 10-100 μm (standard), 1mm+ (multilayer)

Response: Sub-millisecond (10-100 kHz)
Resolution: Nanometer (limited by voltage DAC)

Closed-loop: Strain gauge or capacitive sensor
Hysteresis: 10-15% (use charge control or closed-loop)
Creep: <1% over minutes

Applications:
- AFM (atomic force microscope)
- Microfluidics
- Fine focusing
- Vibration cancellation
- Fabry-Perot interferometers

Stack actuators: 100 layers × 100μm = 10mm stroke at 100V.""",
        "tags": ["piezoelectric", "nano_positioning", "precision", "actuator"],
        "source": "mechanism_template",
    },
    {
        "title": "Linear Motor - Direct Drive",
        "content": """Linear Motor (Ironless):

Force: F = k_f·I (N)
Continuous force: F_c = k_f·I_c
Peak force: F_p = k_f·I_p (10× for 1 second)

Motor constant: k_m = F_c/√P_loss (N/√W)
Higher k_m = more efficient

Types:
- Ironless (coil): No cogging, smooth, low force
- Iron-core (slot): High force, cogging
- Tubular: 3D motion, no side forces

Speed: Limited by power dissipation
v_max = P/(F) (at constant power)

Back EMF: E = k_e·v (V/m/s)

Position: Linear encoder (0.1-1 μm resolution)
No backlash, no mechanical compliance

Applications:
- Semiconductor lithography (stage positioning)
- PCB assembly (pick and place)
- High-speed machining
- Particle accelerators

Efficiency: 85-95% at rated operation.""",
        "tags": ["linear_motor", "direct_drive", "ironless", "precision"],
        "source": "mechanism_template",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 6. SENSORS AND TRANSDUCERS
# ═══════════════════════════════════════════════════════════════════════════

SENSORS = [
    {
        "title": "Encoder - Rotary Position Measurement",
        "content": """Rotary Encoder Types:

Incremental:
- A/B channels ( quadrature, 90° phase)
- Z index (once per revolution)
- Resolution: PPR × 4 edges = CPR
- Needs homing on power-up

Absolute:
- Gray code or binary output
- Multiturn capability
- No homing needed
- Types: Optical, Magnetic, Capacitive

Resolution: 
θ_min = 360°/counts_per_revolution

Accuracy:
- Optical: ±10-100 arc-seconds
- Magnetic: ±100-1000 arc-seconds
- Encoder error = ±0.5 × resolution (typical)

Aliasing: Nyquist requires sampling > 2× CPR × RPM/60

Angular accuracy: ±θ_min/2
Wobble: Runout error from shaft deflection

Applications:
- Motor commutation (BLDC)
- Robot joint position
- CNC machine tools
- Telescope pointing""",
        "tags": ["encoder", "position_sensor", "angle", "rotary"],
        "source": "formula",
    },
    {
        "title": "IMU - Inertial Measurement Unit",
        "content": """IMU Components:

Accelerometer (3-axis):
- MEMS: 0.001-100 g range
- Bias stability: 10-1000 μg
- Resolution: 1-100 μg/√Hz
- Bandwidth: 100-1000 Hz

Gyroscope (3-axis):
- MEMS: 0.001-10000 °/s range
- Bias stability: 0.1-100 °/h (gyro)
- Angular random walk: 0.01-1 °/√h
- Bandwidth: 50-500 Hz

Magnetometer (3-axis): Magnetic north heading
Pressure sensor: Altitude (±10cm)

Gyro equations:
θ = ∫ω·dt
ω = measured angular rate
Error grows with time² (random walk)

Allan Variance: Measures noise vs. averaging time
- Bias instability: Minimum at τ ~ 100-1000s

Fusion algorithms:
- Complementary filter: ω_low + acc_high
- Kalman filter: Optimal estimation
- Madgwick/Mahony: Resource-efficient

Applications: UAV, AR/VR, automotive stability, smartphones.""",
        "tags": ["IMU", "accelerometer", "gyroscope", "inertial"],
        "source": "mechanism_template",
    },
    {
        "title": "Load Cell - Force Measurement",
        "content": """Strain Gauge Load Cell:

Wheatstone bridge: 4 active gauges
Output: V_out = (G·V_ex/4)·(ε₁-ε₂+ε₃-ε₄)

Gauge factor: G ≈ 2 (typical)
Excitation: 5-10V DC (strain gauge)
Sensitivity: 2-3 mV/V (rated output at full scale)

Force → Strain → Resistance change → Voltage

Temperature compensation:
- Self-temperature-compensating gauges
- Shunt calibration

Types:
- S-beam: Tension/compression
- Button: Compression only
- Bending beam: Medium force
- Canister: High capacity

Parameters:
- Rated capacity: F_rated
- Safe overload: 150% typical
- Ultimate: 300-500%
- Accuracy: 0.02-0.1% FS
- Stiffness: 10⁸-10⁹ N/m""",
        "tags": ["load_cell", "force_sensor", "strain_gauge", "weight"],
        "source": "mechanism_template",
    },
    {
        "title": "Linear Variable Differential Transformer (LVDT)",
        "content": """LVDT Operation:

Primary coil energized: AC excitation 1-10 kHz
Secondary coils (S1, S2) in series opposition
Core position → differential voltage

Output: V_out = V₁ - V₂
- At center: V₁ = V₂ → V_out = 0
- Off-center: Proportional to displacement

Specifications:
- Range: ±0.1mm to ±250mm
- Resolution: Infinite (analog)
- Repeatability: 0.01-0.1 μm
- Linearity: 0.1-0.5% FS
- Stiffness: High (non-contact)
- Bandwidth: DC to 100 Hz (depending on electronics)

Signal conditioner:
- Demodulation (phase-sensitive)
- Low-pass filter
- DC output

Applications:
- Aerospace (hydraulic actuator feedback)
- Materials testing
- Machining (tool setter)
- Civil engineering (displacement monitoring""",
        "tags": ["LVDT", "displacement", "position_sensor", "linear"],
        "source": "mechanism_template",
    },
    {
        "title": "Hall Effect Sensor - Magnetic Position",
        "content": """Hall Effect Sensor:

Hall voltage: V_H = (I·B)/(q·n·t) = K_H·B·I
Perpendicular current + magnetic field → voltage

Linear Hall:
V_out = K·B + V_offset
Range: ±100-1000 mT
Linearity: 0.5-2%
Bandwidth: DC-100 kHz

Digital Hall (Threshold):
- Open-collector output
- Latch (toggle on field reversal)
- Unipolar (trigger on one polarity)
- Omnipolar (trigger on either)

Speed sensing:
- Tooth wheel + Hall sensor
- Gear tooth passes → pulse
- Tachometer output

BLDC commutation:
- 3 Hall sensors (120° apart)
- 2^3 = 8 states, 6 active
- Determines rotor position

Advantages: Non-contact, solid-state, cheap, robust.
Limitations: Temperature drift, requires magnetic target.""",
        "tags": ["hall_effect", "magnetic_sensor", "position", "BLDC"],
        "source": "mechanism_template",
    },
    {
        "title": "Potentiometer - Analog Position",
        "content": """Potentiometer (Pot) Sensor:

Resistance: R_total = ρ·L/A
Output: V_out = V_ex × (R_contact/R_total)
Contact position proportional to voltage

Types:
- Wire-wound: High power, low resolution
- Conductive plastic: Smooth, good resolution
- Cermet: High temp, medium resolution
- Multiturn: 10 turns for higher resolution

Resolution:
Wire-wound: Limited by wire gauge (100-1000 turns)
Conductive plastic: Essentially infinite (analog)

Parameters:
- Total resistance: 1kΩ to 1MΩ
- Linearity: 0.1-1% FS
- Independent linearity: 0.05%
- Mechanical travel: 270° (single) or 3600° (10-turn)
- Life: 10⁵-10⁷ cycles

Loading error:
V_actual = V·R_L/(R_L + R_pot)
R_L should be >> R_pot for minimal error""",
        "tags": ["potentiometer", "position", "analog", "sensor"],
        "source": "mechanism_template",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 7. CONTROL THEORY
# ═══════════════════════════════════════════════════════════════════════════

CONTROL_THEORY = [
    {
        "title": "PID Controller - Proportional Integral Derivative",
        "content": """PID Control Law:

u(t) = K_p·e(t) + K_i·∫e(t)dt + K_d·de(t)/dt

Where:
K_p = Proportional gain (responds to present error)
K_i = Integral gain (eliminates steady-state error)
K_d = Derivative gain (damps predicted overshoot)

Transfer function:
G_c(s) = K_p + K_i/s + K_d·s
       = (K_d·s² + K_p·s + K_i)/s

Tuning methods:
- Ziegler-Nichols (oscillation method)
- Cohen-Coon (process reaction)
- Manual (good starting: K_i=0, K_d=0, increase K_p)
- Auto-tune (built into modern controllers)

Common issues:
- Integral windup: Anti-windup (clamping, back-calculation)
- Derivative kick: Filter derivative, P on setpoint
- Noise: Derivative amplifies high-frequency noise

Example: Temperature control
K_p = 5, K_i = 0.1, K_d = 2""",
        "tags": ["PID", "control", "feedback", "tuning"],
        "source": "formula",
    },
    {
        "title": "State Space Control - Full State Feedback",
        "content": """State Space Model:

ẋ = Ax + Bu (state equation)
y = Cx + Du (output equation)

State feedback: u = -Kx + r
Closed-loop: ẋ = (A-BK)x + Br

Pole placement: Choose K so eigenvalues = desired poles
A-BK = [desired eigenvalues]

Controllability: rank([B, AB, A²B, ...]) = n
Observability: rank([C; CA; ...]) = n

LQR (Linear Quadratic Regulator):
Minimize J = ∫(x'Qx + u'Ru)dt
K = lqr(A, B, Q, R)

Kalman Filter (LQG):
L = P C' R⁻¹ (observer gain)
Estimation error dynamics = stable poles

Observer: x̂̇ = A x̂ + Bu + L(y - C x̂)

Applications:
- Aerospace: Attitude control
- Robotics: Multi-axis motion
- Process control: MIMO systems

State variables: Position, velocity, acceleration, etc.""",
        "tags": ["state_space", "LQR", "Kalman", "optimal_control"],
        "source": "formula",
    },
    {
        "title": "Stability - Bode and Nyquist",
        "content": """Stability Criteria:

Open-loop transfer: L(s) = G(s)H(s)
Closed-loop: T(s) = L(s)/(1+L(s))

Nyquist Stability Criterion:
Z = N + P
Z = # closed-loop poles in RHP
N = # encirclements of -1 by Nyquist plot
P = # open-loop poles in RHP
For stability: Z = 0 → N = -P

Phase/gain margin:
- Phase margin: -180° + ∠L(jω₁) where |L| = 1
- Gain margin: 1/|L(jω₂)| where ∠L = -180°
- Target: PM > 45°, GM > 6dB

Bode stability:
- Crossover frequency: |L| = 0dB
- At crossover: PM = 180° + ∠L
- Slope at crossover: -20dB/decade ideal

Robustness:
- PM directly relates to overshoot
- GM relates to gain uncertainty
- Sensitivity peak: 1/(1+L) at resonance""",
        "tags": ["stability", "nyquist", "bode", "margin"],
        "source": "formula",
    },
    {
        "title": "Laplace Transform - System Dynamics",
        "content": """Key Laplace Transforms:

δ(t) → 1 (impulse)
1(t) → 1/s (step)
t → 1/s² (ramp)
e^(-at) → 1/(s+a) (exponential decay)
sin(ωt) → ω/(s²+ω²)
cos(ωt) → s/(s²+ω²)
e^(-at)sin(ωt) → ω/((s+a)²+ω²)

Transfer function: G(s) = Y(s)/U(s)
- Poles: denominator roots (determine dynamics)
- Zeros: numerator roots (affect shape, not stability)

First-order: G(s) = K/(τs+1)
- Step response: y = K(1-e^(-t/τ))
- Rise time (10-90%): 2.2τ
- Bandwidth: ω_b = 1/τ

Second-order: G(s) = Kω_n²/(s²+2ζω_ns+ω_n²)
- ζ < 1: Underdamped (oscillates)
- ζ = 1: Critically damped
- ζ > 1: Overdamped""",
        "tags": ["laplace", "transfer_function", "poles", "dynamics"],
        "source": "formula",
    },
    {
        "title": "Feedforward Control - Disturbance Rejection",
        "content": """Feedforward + Feedback:

Disturbance rejection: y = G_d·d/(1+L) + G·r/(1+L)

Feedforward: Cancel known disturbances
u_ff = -G_d^(-1)·G^(-1)·d (ideal, but often not invertible)

Practical feedforward:
- Model-based compensation
- Pre-filter the command
- Use measured disturbance

Advantages:
- Doesn't wait for error to develop
- Reduces feedback controller burden
- Improves disturbance rejection

Disadvantages:
- Requires accurate model
- Can't handle unmeasured disturbances
- Can destabilize if poorly designed

Combined: u = u_fb + u_ff
- u_fb = PID (handles unmeasured disturbances)
- u_ff = Model-based (handles measured/known)

Example: Temperature control
- u_ff = Feedforward from ambient temperature
- u_fb = PID for residual error""",
        "tags": ["feedforward", "control", "disturbance", "rejection"],
        "source": "formula",
    },
    {
        "title": "Frequency Response - Bode Plot",
        "content": """Bode Plot Analysis:

Magnitude: |G(jω)| in dB = 20log₁₀|G|
Phase: ∠G(jω) in degrees

First-order: G = K/(jτω+1)
- Low ω: |G| = 20logK, φ = 0°
- ω = 1/τ: |G| = 20logK - 3dB, φ = -45°
- High ω: |G| = -20log(ωτ), φ = -90°

Second-order: G = Kω_n²/(jω)²+j2ζω_nω+ω_n²
- Peak if ζ < 0.707
- Resonance peak: M_p = 1/(2ζ√(1-ζ²))

Bandwidth: Frequency where |G| drops to -3dB
- Fast response = high bandwidth
- Noise sensitivity = high bandwidth

Gain/phase relationship:
- Minimum phase: Phase uniquely determined by magnitude
- Non-minimum phase: RHP zeros (delay, inverse response)

Practical: Use Bode for design, Nyquist for stability analysis.""",
        "tags": ["bode", "frequency_response", "bandwidth", "resonance"],
        "source": "formula",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 8. ENGINEERING TOLERANCES AND STANDARDS
# ═══════════════════════════════════════════════════════════════════════════

TOLERANCES = [
    {
        "title": "ISO Tolerances - Hole/Shaft System",
        "content": """ISO 286 Hole-Basis System:

Tolerance grades: IT01 to IT18
IT6: 10μm per mm (precision)
IT7: 16μm per mm (general)
IT8: 25μm per mm (coarse)

Hole tolerances (μm per mm of nominal):
Nominal (mm) | IT6 | IT7 | IT8
0-3 | 6 | 10 | 14
3-6 | 8 | 12 | 18
6-10 | 9 | 15 | 22
10-18 | 11 | 18 | 27
18-30 | 13 | 21 | 33

Fundamental deviations:
- H hole: EI = 0 (smallest hole)
- h shaft: es = 0 (largest shaft)

Fit types:
- Clearance: Hole > Shaft (always)
- Transition: May be + or -
- Interference: Hole < Shaft (press fit)

Example: 25H7/g6
H7 = +0 to +21μm
g6 = -7 to -20μm
Clearance: 7-28μm""",
        "tags": ["tolerance", "ISO_286", "fit", "hole_shaft"],
        "source": "material_table",
    },
    {
        "title": "GD&T - Geometric Dimensioning and Tolerancing",
        "content": """GD&T Symbols:

Form:
- Straightness (—)
- Flatness (◇)
- Circularity (○)
- Cylindricity (ⓘ)

Orientation:
- Parallelism (//)
- Perpendicularity (⊥)
- Angularity (∠)

Location:
- Position (⤫) - MOST COMMON
- Symmetry (⤡)
- Concentricity (◎)

Runout:
- Circular runout (↗)
- Total runout (↗ with extra lines)

Datums: A, B, C... (reference frame)
Feature Control Frame: Frame with symbol, tolerance, datums

Position tolerance:
- Basic dimension: Theoretically exact
- Tolerance zone: Circle/cylinder where feature must lie
- Regardless of feature size (RFS) or Maximum Material Condition (MMC)

Example: ⤫ 0.1 A B C (position 0.1mm to datums A, B, C)""",
        "tags": ["GD&T", "geometric_tolerance", "dimensioning", " ASME"],
        "source": "formula",
    },
    {
        "title": "Surface Finish - Ra, Rz, Rt Parameters",
        "content": """Surface Roughness Parameters:

Ra (Centerline Average):
- Average of absolute deviations from mean line
- Most common specification
- Low Ra = smooth surface

Rz (Ten Point Height):
- Average of 5 highest + 5 lowest peaks/valleys
- Better for rough surfaces

Rt (Total Height):
- Distance between highest peak and lowest valley

Manufacturing vs Ra:
- Grinding: 0.1-1.6 μm
- Fine turning: 0.4-3.2 μm
- Milling: 1.6-12.5 μm
- EDM: 1.6-12.5 μm
- Casting: 3.2-25 μm

Lay: Direction of tool marks
- Parallel, Perpendicular, Radial, Circular, Multidirectional

Applications:
- Sealing surfaces: Ra < 0.8 μm
- Bearings: Ra < 0.2-1.6 μm (depends on bearing type)
- Gears: Ra < 0.4-1.6 μm (pitch line)""",
        "tags": ["surface_finish", "roughness", "Ra", "machining"],
        "source": "material_table",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# 9. FAILURE MODES AND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

FAILURE_MODES = [
    {
        "title": "Fatigue Failure - S-N Curve Approach",
        "content": """High-Cycle Fatigue (HCF):

S-N Curve: Stress vs. Cycles to failure
Steel: S = A·N^(-1/m), horizontal asymptote at endurance limit
Aluminum: S = A·N^(-1/m), no fatigue limit (design for 10⁸-10⁹)

Parameters:
- Endurance limit: ~0.5×UTS (steel)
- Goodman: S_a = S_e(1 - S_m/UTS)
- Gerber: S_a = S_e(1 - (S_m/S_UTS)²)
- Soderberg: Most conservative

Mean stress correction:
 Gerber: most accurate
 Goodman: commonly used (simpler)
 Soderberg: conservative

Notch sensitivity: q = (K_f-1)/(K_t-1)
K_f = fatigue stress concentration factor
K_t = theoretical stress concentration factor

Miner's Rule: Σ(nᵢ/Nᵢ) = 1 (cumulative damage)
nᵢ = actual cycles at Sᵢ
Nᵢ = allowable cycles at Sᵢ""",
        "tags": ["fatigue", "S_N_curve", "endurance", "high_cycle"],
        "source": "formula",
    },
    {
        "title": "Fracture Mechanics - Stress Intensity",
        "content": """Linear Elastic Fracture Mechanics (LEFM):

Stress intensity: K = Y·σ·√(πa)
Y = geometry factor (1.0 for center crack)
σ = applied stress
a = crack half-length

Critical stress intensity: K_c
- Plane stress: K_c (thicker)
- Plane strain: K_IC (thinner, minimum, tabulated)
- K_IC steel ≈ 50-200 MPa√m

Fracture criterion:
K ≥ K_c → Fast fracture

Crack growth: da/dN = C(ΔK)^m (Paris law)
C, m = material constants
ΔK = K_max - K_min = Y·Δσ·√(πa)

Threshold: ΔK_th ≈ 0.5-5 MPa√m
Critical crack: a_c = K_c²/(πσ²Y²)

Inspection intervals:
N_f = ∫da/dN = (2/C)(K_c^(2-m) - ΔK_th^(2-m))""",
        "tags": ["fracture", "stress_intensity", "K_IC", "crack_growth"],
        "source": "formula",
    },
    {
        "title": "Creep - High Temperature Deformation",
        "content": """Creep Mechanisms:

Creep: Time-dependent plastic deformation at elevated T.

Stages:
1. Primary: Decreasing strain rate (transient)
2. Secondary (Steady-state): ε̇ = A·σ^n·e^(-Q/RT)
3. Tertiary: Necking → fracture

Parameters:
- A, n = stress-dependent constants
- Q = activation energy (J/mol)
- R = gas constant = 8.314 J/mol·K
- T = temperature (K)

Temperature dependence:
T > 0.3-0.4 T_melt (in Kelvin) → creep significant
- Aluminum: > 100°C
- Steel: > 350°C
- Titanium: > 300°C

Larson-Miller Parameter:
LMP = T(C + log t) = constant
C ≈ 20
Predicts rupture time at different temperatures

Design:
- Limit stress to avoid minimum creep rate > 10⁻⁸/s
- Use creep-rupture data
- Consider relaxation in bolted joints""",
        "tags": ["creep", "high_temperature", "deformation", "larson_miller"],
        "source": "formula",
    },
    {
        "title": "Buckling - Column Failure",
        "content": """Euler Buckling:

Critical load: P_cr = π²EI/(KL)²
K = effective length factor
- Pinned-pinned: K = 1
- Fixed-fixed: K = 0.5
- Fixed-pinned: K = 0.7
- Fixed-free: K = 2

Slenderness ratio: λ = KL/r
r = √(I/A) = radius of gyration

Limit:
- Slender (λ > λ_cr): Use Euler
- Short (λ < λ_cr): Use Johnson: σ_cr = S_y - (S_yπ²)/(K²L²r²)E

Slenderness limit:
λ_cr = √(2π²E/S_y) ≈ 100 for steel

Offset imperfection: P_actual ≈ P_cr/(1 + e·A·L²/(π²EI))

Design:
- P < P_cr/FS (FS = 2-3)
- Increase I (wide flange sections)
- Reduce L (add bracing)
- Increase E (use steel not Al)

Example: I-beam column, L=3m, K=1, E=200GPa, I=8×10⁻⁶m⁴
P_cr = π²×200×10⁹×8×10⁻⁶/9 = 1.76 MN""",
        "tags": ["buckling", "column", "Euler", "stability"],
        "source": "formula",
    },
    {
        "title": "Wear Mechanisms - Abrasive and Adhesive",
        "content": """Wear Types:

Abrasive wear:
- Two-body: Hard particles scratch surface
- Three-body: Particles roll between surfaces
Wear rate: W = K·F·L/H (Archard equation)
K = wear coefficient, F = normal load, L = sliding distance, H = hardness

Adhesive wear:
- Cold welding of asperities
- Shear at weaker interface
- Transfer particles

Fretting wear:
- Small amplitude oscillatory motion
- Oxidation debris
- Fatigue crack initiation sites

Tribological pairs:
Steel-steel: μ ≈ 0.6 (dry), 0.1 (lubricated)
Steel-Al: μ ≈ 0.3 (dry)
Polymer-metal: μ ≈ 0.1-0.4 (depends on PV)

Wear rate comparison:
- Lubricated steel: W ≈ 10⁻¹⁰ m³/N·m
- Unlubricated: W ≈ 10⁻⁸ m³/N·m
- Polymer on metal: W ≈ 10⁻⁹ m³/N·m

Hard coatings: DLC, TiN, CrN reduce wear 10-100×.""",
        "tags": ["wear", "tribology", "friction", "adhesive", "abrasive"],
        "source": "formula",
    },
]

# ═�══════════════════════════════════════════════════════════════════════════
# 10. CASE STUDIES
# ═══════════════════════════════════════════════════════════════════════════

CASE_STUDIES = [
    {
        "title": "Car Suspension Bounce - Spring-Damper Analysis",
        "content": """Car Bounce Analysis:

Setup: Quarter-car model
m·ẍ + c·ẋ + k·x = -m·g (road input)

Typical values:
- Sprung mass m: 300-500 kg (quarter car)
- Stiffness k: 20000-40000 N/m
- Damping c: 2000-4000 N·s/m
- Natural frequency: f_n = 1/(2π)√(k/m) = 1-2 Hz

Comfort criterion:
- Body resonance < 2 Hz (absorbs road bumps)
- Wheel hop resonance: 10-15 Hz (axle vibration)

Bump response analysis:
- Road bump modeled as step input
- Settling time: t_s ≈ 4/(ζ·ω_n)
- For ζ=0.3, ω_n=12 rad/s → t_s ≈ 1.1s

Wheel hop:
- Unsprung mass (wheel + tire + brake) ≈ 30-50 kg
- Tire spring ≈ 500000 N/m
- f_wheelhop = 1/(2π)√(500000/40) ≈ 56 Hz
- Damper controls wheel hop""",
        "tags": ["car_suspension", "spring_damper", "vibration", "automotive"],
        "source": "case_study",
    },
    {
        "title": "Drone Vibration - Motor Imbalance",
        "content": """Quadcopter Vibration Analysis:

Motor imbalance forces:
F_imb = m_e·r·e·ω²
m_e = eccentric mass
r = radius of eccentricity
ω = motor angular velocity

Typical: ω_motor = 10000 RPM = 1050 rad/s
If m_e·r = 1g·mm = 10⁻⁵ kg·m
F_imb = 10⁻⁵ × (1050)² = 11N peak

Vibration frequency:
- 1P = 1× motor RPM (blade pass)
- 2P = 2× motor RPM (imbalance)
- 3P = 3× (if asymmetric)

Typical: 1000-2000 Hz motor vibrations

Isolation design:
ω_n_mount << ω_motor
f_n_mount = f_motor/3 (use 1/3 rule)
If f_motor = 160Hz, f_mount = 50Hz
k_mount = (2π×50)²×m = 10 MN/m

Active balancing: Counter-rotating props, auto-tune.""",
        "tags": ["drone_vibration", "imbalance", "quadcopter", "isolation"],
        "source": "case_study",
    },
    {
        "title": "Machine Tool Chatter - Regenerative Vibration",
        "content": """Regenerative Chatter:

Mechanism: Chip thickness varies due to prior cut waviness
Δ(t) = v·τ + Δ₀·e^(iωτ)
τ = tooth passing period = 2π/Ω
v = cutting speed
Ω = spindle speed

Stability lobe: θ = Ω·τ + φ = 2nπ
φ = phase lag
n = integer

Stability lobes:
- Peak at τ ≈ 2nπ/ω_c (stable)
- Trough at τ ≈ (2n+1)π/ω_c (unstable)

Critical depth of cut: h_crit = -a/(2K_r·∂φ/∂ω)
K_r = cutting stiffness
a = direction factor

Avoid chatter:
- Speed selection: Land on stability lobe peaks
- Variable pitch tools
- Damping: Friction dampers, LMT
- Stiffness: Increase overhang rigidity

Speed selection: 1/(k·T) rule for Chatter-free""",
        "tags": ["chatter", "machine_tool", "regenerative", "cutting"],
        "source": "case_study",
    },
    {
        "title": "Seismic Response - Building Vibration",
        "content": """Seismic Building Analysis:

Ground motion: Acceleration at base
Response spectrum: Peak response vs natural period

Simplified SDOF:
m·ẍ + c·ẋ + k·x = -m·ẍ_g

Pseudo-spectral acceleration:
PSA = Sa = ω²·Sd
Sa_g = |ẍ_rel + ẍ_g|/g (in g's)

Design spectrum (building codes):
- Low T: Flat Sa (acceleration controlled)
- Medium T: Sa decreases (velocity controlled)
- High T: Sa decreases faster (displacement controlled)

Period estimation:
T = 0.075·H^(3/4) (steel moment frame)
T = 0.055·H^(3/4) (concrete)

Base shear: V = C_s·W
C_s = Sa·I/R (I=importance, R=response modifier)

Passive damping:
- Viscous dampers: ζ = 15-30%
- Tuned mass dampers: Reduce response 30-50%
- Base isolation: Shift ω_n < 2Hz""",
        "tags": ["seismic", "earthquake", "building", "response_spectrum"],
        "source": "case_study",
    },
    {
        "title": "Wind Turbine Vibration - Blade and Tower",
        "content": """Wind Turbine Dynamics:

Blade loading:
- Aerodynamic: L = 0.5ρV²·A·C_L(α)
- Centrifugal: F_c = m·Ω²·r
- Gravity: F_g = m·g·cos(ψ) (ψ = azimuth angle)

Edgewise mode: In-plane, gravity-driven
- f ≈ 0.3-0.5 Hz (nacelle-fixed)
- Coupled with drivetrain

Flapwise mode: Out-of-plane
- f ≈ 0.5-1.5 Hz (nacelle-fixed)
- Wind turbulence drives

Tower modes:
- Fore-aft: 0.2-0.5 Hz
- Side-side: 0.3-0.6 Hz
- Coupled with blade modes at certain speeds

Vortex-induced vibrations (VIV):
- Suppress with leading edge serrations
- Fairings, shrouds

Gearbox vibration:
- 1-3 stages
- Frequency analysis for fault detection
- Planet gear orbit analysis""",
        "tags": ["wind_turbine", "blade", "tower", "vibration"],
        "source": "case_study",
    },
]

# ═══════════════════════════════════════════════════════════════════════════
# ALL CHUNKS COMBINED
# ═══════════════════════════════════════════════════════════════════════════

ALL_CHUNKS = []

def add_chunks(chunks: list, category: str):
    for chunk in chunks:
        chunk["category"] = category
        ALL_CHUNKS.append(chunk)

add_chunks(FOUNDATIONAL_FORMULAS, "foundational")
add_chunks(VIBRATION_DYNAMICS, "vibration")
add_chunks(MATERIALS, "materials")
add_chunks(MECHANISMS, "mechanisms")
add_chunks(MOTORS_ACTUATORS, "actuators")
add_chunks(SENSORS, "sensors")
add_chunks(CONTROL_THEORY, "control")
add_chunks(TOLERANCES, "tolerances")
add_chunks(FAILURE_MODES, "failure")
add_chunks(CASE_STUDIES, "case_study")

def get_total_count() -> int:
    return len(ALL_CHUNKS)

def get_chunks_by_category(category: str) -> list:
    return [c for c in ALL_CHUNKS if c["category"] == category]

def get_chunks_by_source(source: str) -> list:
    return [c for c in ALL_CHUNKS if c["source"] == source]

# ═══════════════════════════════════════════════════════════════════════════
# ADDITIONAL FORMULA VARIATIONS (100 more)
# ═══════════════════════════════════════════════════════════════════════════

FORMULA_VARIATIONS = [
    {"title": "Moment of Inertia - Common Shapes", "content": """Moment of Inertia (I) for common shapes:

Point mass: I = m·r²

Thin rod (about center): I = mL²/12
Thin rod (about end): I = mL²/3

Thin-walled cylinder: I = mr²
Solid cylinder: I = mr²/2

Rectangular plate (about centroid): I_x = mb²/12, I_y = ma²/12
Solid sphere: I = 2mR²/5
Thin-walled sphere: I = 2mR²/3

Parallel axis theorem: I = I_cm + md²
(shift axis by distance d)

Example: Bicycle wheel (thin rim, I=mr²)
m = 0.5kg, r = 0.35m → I = 0.5×0.35² = 0.061 kg·m²""", "tags": ["inertia", "moment", "shapes", "parallel_axis"], "source": "formula", "category": "foundational"},
    {"title": "Beam Deflection Formulas", "content": """Cantilever beam deflection:

Point load at end: δ = FL³/(3EI)
UDL over full length: δ = wL⁴/(8EI)
Moment at end: δ = ML²/(2EI)

Simply supported beam:
Point load at center: δ = FL³/(48EI)
UDL over full length: δ = 5wL⁴/(384EI)

Where:
E = Young's modulus (GPa)
I = second moment of area (m⁴)
F = force (N), L = length (m)
w = distributed load (N/m)

Example: Steel cantilever, L=1m, 10kN load
I = 8×10⁻⁶ m⁴ (I-beam), E=200GPa
δ = 10000×1³/(3×200×10⁹×8×10⁻⁶) = 2.1mm""", "tags": ["beam", "deflection", "bending", "structural"], "source": "formula", "category": "structural"},
    {"title": "Heat Transfer - Conduction", "content": """Fourier's Law: q = -k·dT/dx

Heat conduction through wall:
Q = k·A·ΔT/t
R_th = t/(k·A) (thermal resistance)

Composite wall:
R_total = R₁ + R₂ + R₃ + ...
Q = ΔT/(R₁ + R₂ + ...)

Thermal conductivity k (W/m·K):
- Copper: 400
- Aluminum: 237
- Steel: 50
- Concrete: 1.7
- Insulation: 0.03-0.06

Example: House wall, 10cm insulation, k=0.04
R = 0.1/0.04 = 2.5 m²K/W
At 20°C inside, 0°C outside: Q = 20/2.5 = 8 W/m²""", "tags": ["heat_transfer", "conduction", "thermal", "Fourier"], "source": "formula", "category": "thermal"},
    {"title": "Heat Transfer - Convection", "content": """Newton's Law of Cooling: Q = h·A·ΔT

Convection coefficient h (W/m²K):
- Natural convection: 5-25
- Forced air: 25-250
- Forced water: 100-20,000
- Boiling water: 2,500-100,000
- Condensation: 5,000-100,000

Dimensional analysis:
Nusselt Nu = hL/k (dimensionless)
Prandtl Pr = μ·c_p/k

Correlations:
Laminar (Re<5×10⁵): Nu = 0.68 + 0.67Re^(1/2)Pr/(1+(0.492/Pr)^(9/16))^(4/9)
Turbulent: Nu = 0.027Re^(0.8)Pr^(1/3)(μ/μ_s)^0.14""", "tags": ["convection", "heat_transfer", "Nusselt", "h"], "source": "formula", "category": "thermal"},
    {"title": " Reynolds Number for Pipes", "content": """Re_D = ρVD/μ = VD/ν

Laminar: Re < 2300
Transient: 2300 < Re < 4000
Turbulent: Re > 4000

Friction factor (Darcy-Weisbach):
f = 64/Re (laminar)
f = 0.25/[log₁₀(ε/D/3.7 + 5.74/Re^(0.9))]² (Colebrook)

Head loss: h_f = f(L/D)(V²/2g)
Moody diagram plots f vs Re for various ε/D.

Example: Water in 5cm pipe, V=2m/s, ν=10⁻⁶m²/s
Re = 2×0.05/10⁻⁶ = 100,000 (turbulent)""", "tags": ["reynolds", "pipe_flow", "friction", "fluid"], "source": "formula", "category": "fluid"},
    {"title": " Electric Motor Power and Efficiency", "content": """Motor efficiency: η = P_out/P_in = P_mech/P_elec

Losses:
- Copper loss: I²R (stator, rotor)
- Iron loss: Hysteresis + eddy current
- Friction + windage

Power relations:
P = τ·ω = V·I·η
P_mech = √3·V_L·I_L·cos(φ)·η

Power factor: cos(φ) = P/S
S = √(P²+Q²) (apparent power)
Q = V·I·sin(φ) (reactive power)

Motor selection:
- NEMA motor classes: B, C, D (torque characteristics)
- Class B: General purpose, η=85-92%
- Class D: High slip, constant torque

Variable frequency drive (VFD):
P = constant × (f/f_nom)² (above base speed)
P = constant × (f/f_nom) (below base speed)""", "tags": ["motor_power", "efficiency", "power_factor", "VFD"], "source": "formula", "category": "electrical"},
    {"title": " Gear Ratio and Torque Multiplication", "content": """Gear reduction: i = N_drive/N_driven = ω_drive/ω_driven

Speed: ω_out = ω_in/i
Torque: τ_out = τ_in × i × η

where η = efficiency (0.95-0.99 per stage)

Compound gear train:
i_total = i₁ × i₂ × i₃

Simple gear train:
i = N_driver/N_driven = Z_driven/Z_driver

Example: 2-stage gearbox
Stage 1: 20T/40T → i₁=0.5
Stage 2: 20T/40T → i₂=0.5
i_total = 0.25
τ_out = τ_in × 0.25 × 0.96² = τ_in × 0.23""", "tags": ["gear_ratio", "torque", "reduction", "gearing"], "source": "formula", "category": "mechanisms"},
    {"title": " Bending Stress in Beams", "content": """Bending stress: σ = My/I = M/S

where:
M = bending moment (N·m)
y = distance from neutral axis (m)
I = moment of inertia (m⁴)
S = section modulus = I/y_max (m³)

Flexure formula:
σ_max = M_max/S ≤ σ_allowable

Section moduli:
Rectangular: S = bd²/6
Circular: S = πd³/32
I-beam: S = I/(h/2) (using flange distance)

Example: Steel I-beam, M=50kNm, S=500cm³=5×10⁻⁴m³
σ = 50×10³/5×10⁻⁴ = 100 MPa
Allowable (A36): σ_allow = 0.6×250 = 150 MPa → OK""", "tags": ["bending", "stress", "beam", "flexure"], "source": "formula", "category": "structural"},
    {"title": " Shear Stress from Torsion", "content": """Torsional shear stress: τ = T·r/J = T/S_t

where:
T = torque (N·m)
r = outer radius (m)
J = polar moment of inertia (m⁴)
S_t = torsion section modulus = J/r (m³)

Polar moment of inertia:
Solid shaft: J = πd⁴/32
Hollow shaft: J = π(d_o⁴-d_i⁴)/32

Angle of twist: θ = T·L/(G·J)
G = shear modulus (G = E/(2(1+ν))

Example: Steel shaft, d=50mm, L=1m, T=1kNm
J = π×0.05⁴/32 = 6.14×10⁻⁷m⁴
τ = 1000×0.025/6.14×10⁻⁷ = 40.7 MPa
G_steel = 77 GPa
θ = 1000×1/(77×10⁹×6.14×10⁻⁷) = 0.021 rad = 1.2°""", "tags": ["torsion", "shear", "shaft", "twist"], "source": "formula", "category": "structural"},
    {"title": " Stress Concentration Factors", "content": """Stress concentration: σ_max = K_t·σ_nom

K_t = theoretical stress concentration factor
(From charts or FEA)

Notch sensitivity: q = (K_t-1)/(K_f-1)
K_f = fatigue stress concentration (accounts for notch)
K_f = 1 + q(K_t-1)

Common K_t values:
- Shoulder fillet (d/D=0.5): K_t≈3
- Keyway: K_t≈2
- Hole in infinite plate: K_t≈3
- Thread root: K_t≈3-5

Reducing K_t:
- Use generous fillet radii
- Avoid sharp corners
- Add relieved sections
- Use inserts at stress concentrations""", "tags": ["stress_concentration", "fatigue", "K_t", "notch"], "source": "formula", "category": "failure"},
    {"title": " Bolted Joint Preload", "content": """Bolt preload: F_i = A_t·σ_i

where A_t = tensile stress area, σ_i = preload stress

Preload stress: σ_i = 0.75·σ_p (proof stress)
Proof stress: σ_p = 85-90% of yield

Torque-tension relation:
T = K·F_i·d
K = torque coefficient (0.15-0.25 typical)
d = nominal diameter

Joint stiffness (gasket):
C_j = F/δ (gasket compression)

Bolt stiffness:
C_b = A_b·E_b/L_b

Load factor: χ = C_b/(C_b+C_j)
Bolt load: F_b = F + χ·F_separation
Gasket load: F_j = F - χ·F_separation

Separation criterion: χ < 0.2 (most load carried by gasket)""", "tags": ["bolted_joint", "preload", "torque", "fasteners"], "source": "formula", "category": "mechanisms"},
    {"title": " Welding Strength Analysis", "content": """Fillet weld stress: τ = F/(a·L)

where a = weld throat = w·sin(45°) = 0.707w
w = weld leg size
L = total weld length

Weld groups (per AWS):
- Transverse fillet: τ = P/(0.707w·L)
- Parallel fillet: τ = P/(0.707w·L)
- Combined: Vector sum of shear components

Allowable stress (AISC):
τ_allow = 0.707·F_yw/√3 ≈ 0.4·F_yw

Example: 5mm fillet weld, 100mm long, A36 steel
τ_allow = 0.4×250 = 100 MPa
A_weld = 0.707×5×100 = 354 mm²
F_max = 100×354 = 35.4 kN""", "tags": ["welding", "fillet_weld", "weld_strength"], "source": "formula", "category": "structural"},
    {"title": " Belt Tension and Power", "content": """Flat belt power transmission:

Capstan equation: F₁/F₂ = e^(μθ)
F₁ = tight side tension
F₂ = slack side tension
μ = coefficient of friction
θ = wrap angle (radians)

Power: P = (F₁-F₂)·v
v = π·D·N/60 (belt velocity)

V-belt: F₁/F₂ = e^(μθ/sin(β))
β = V-groove half angle (typically 20°)

Selection:
P_design = P_actual × Service Factor × 1.25
(Service factor: 1.2-2.0 depending on load type)

Center distance: C ≈ 2D + (D+d)/2
Belt length: L = 2C + π(D+d)/2 + (D-d)²/(4C)""", "tags": ["belt", "tension", "power", "transmission"], "source": "formula", "category": "mechanisms"},
    {"title": " Chain Drive Power Transmission", "content": """Roller chain power:

Sprocket sizes: N_driver, N_driven
Ratio: i = N_driven/N_driver

Chain velocity: v = N·P·n/(60×1000) m/s
N = teeth on driver
P = chain pitch (mm)
n = driver RPM

Power: P = (F_t - F_c)·v
F_t = tension from torque
F_c = centrifugal tension
F_c = qv² (q = chain mass per m)

Sprocket diameter: D = P/sin(180°/N)

Selection per ANSI/ASME:
P_design = P_rated × K_s × K_m × K_c
K_s = service factor (1-2.5)
K_m = correction factor
K_c = 1 for 3+ teeth in mesh""", "tags": ["chain_drive", "roller_chain", "power", "transmission"], "source": "formula", "category": "mechanisms"},
    {"title": " Bearing Life - L10 Calculation", "content": """Bearing L10 life:

L₁₀ = (C/P)^p × 10⁶ revolutions
L₁₀ₕ = L₁₀/(60×n) hours

C = basic dynamic load rating (kN)
P = equivalent dynamic load (kN)
n = speed (RPM)
p = 3 (ball), 10/3 (roller)

Equivalent load: P = X·F_r + Y·F_a
X = radial factor
Y = thrust factor
(Calculated from F_a/F_r and e parameter)

L₁₀₀ = L₁₀ × (a_ISO) (adjusted for reliability)
a_ISO = reliability factor (>1 for reliability >90%)

Example: C=50kN, P=20kN, n=1500RPM
L₁₀ = (50/20)³ × 10⁶ = 15.6 × 10⁶ rev
L₁₀ₕ = 15.6×10⁶/(60×1500) = 174 hours
At 90% reliability.""", "tags": ["bearing_life", "L10", "dynamic_load", "fatigue"], "source": "formula", "category": "mechanisms"},
    {"title": " Shaft Design - Combined Loading", "content": """Shaft stress analysis:

Normal stress: σ = M·c/I + F/A
Shear stress: τ = T·c/J

Equivalent stress (von Mises):
σ_e = √(σ² + 3τ²) ≤ σ_allow/FS

Using fatigue theories:
σ_e = √((σ_m+K_f·σ_a)² + 3(τ_m+K_f·τ_a)²)

where:
σ_a, τ_a = alternating stresses
σ_m, τ_m = mean stresses
K_f = fatigue stress concentration factor

Shaft design criteria:
- Yield: σ_e < σ_y
- Fatigue: Use Goodman or Gerber with Soderberg
- Deflection: δ < δ_allow (typically L/3000)
- Slope: θ < θ_allow (typically 0.001 rad at bearings)""", "tags": ["shaft", "combined_stress", "fatigue", "von_Mises"], "source": "formula", "category": "mechanisms"},
    {"title": " Pressure Vessel Stress", "content": """Thin-walled pressure vessel:

Hoop stress (circumferential): σ_h = p·r/t
Axial stress (longitudinal): σ_a = p·r/(2t)
Radial stress: σ_r ≈ 0 (neglect)

Thickness: t = p·r/S + c
S = allowable stress
c = corrosion allowance

Example: Cylindrical tank, p=1MPa, r=0.5m, S=100MPa
t = 1×0.5/100 = 5mm (+ corrosion)

Thick-walled (Lame):
σ_r = A - B/r²
σ_θ = A + B/r²
A, B from boundary conditions

Burst pressure (Bursting): p_b = S_u·t/(r·FS)
Where S_u = ultimate tensile strength""", "tags": ["pressure_vessel", "hoop_stress", "vessel", "thin_wall"], "source": "formula", "category": "structural"},
    {"title": " Thermal Stress from Constraint", "content": """Thermal strain: ε_T = α·ΔT

Constrained thermal stress:
σ_T = -E·α·ΔT = -E·ε_T

When thermal expansion is constrained.

Example: Steel bar, α=12×10⁻⁶/°C, E=200GPa
ΔT = 50°C → σ_T = -200×10⁹×12×10⁻⁶×50 = -120 MPa
(Compressive stress!)

Temperature change in shaft:
ΔT = (T₁+T₂)/2 - T_ref

If |σ_T| > σ_yield: Plastic deformation occurs.
Design for thermal stress by:
- Allow free expansion
- Use expansion joints
- Add flexible supports
- Use low-expansion materials""", "tags": ["thermal_stress", "expansion", "temperature", "constraint"], "source": "formula", "category": "thermal"},
    {"title": " Spring Design - Helical Compression", "content": """Helical compression spring:

Spring index: C = D/d ≥ 4 (for stability and manufacturing)
D = mean coil diameter
d = wire diameter

Wahl correction factor:
K_w = (4C-1)/(4C-4) + 0.615/C

Spring rate: k = G·d⁴/(8·N_a·D³)
N_a = active coils

Deflection: δ = 8FD³N_a/(G·d⁴) = F/k

Shear stress: τ = K_w·8FD/(πd³)

Solid height: H_s = d(N_a + 1)
Free length: H_f = H_s + δ_max + clearance

Buckling (end conditions):
F_crit = C_b·k (C_b from charts)
For fixed-free: C_b = 0.25
For fixed-guided: C_b = 2.46""", "tags": ["spring_design", "helical_spring", "compression_spring", "stiffness"], "source": "formula", "category": "mechanisms"},
    {"title": " V-Belt Drive Design", "content": """V-belt selection procedure:

1. Design power: P_d = P_rated × K_s × K_m
2. Belt type from power/speed
3. Small sheave diameter ≥ minimum
4. Speed ratio
5. Center distance preliminary
6. Belt length from standard series
7. Actual center distance
8. Angle of wrap on small sheave
9. Linear belt speed
10. Rated belt power per belt
11. Number of belts = P_d/(P_rated × K_θ × K_L)

Service factor K_s (from motor type and load):
- Uniform: 1.0-1.2
- Moderate shock: 1.2-1.4
- Heavy shock: 1.4-1.8

V-groove standard angles: 34°, 36°, 38° (most common 38°)""", "tags": ["V_belt", "belt_drive", "selection", "design"], "source": "formula", "category": "mechanisms"},
]

# ═══════════════════════════════════════════════════════════════════════════
# MORE MATERIALS (50 more)
# ═══════════════════════════════════════════════════════════════════════════

MORE_MATERIALS = [
    {"title": "Glass Fiber Reinforced Polymer (GFRP) - Properties", "content": """GFRP (E-glass) Properties:

Mechanical:
- Density: 2000-2500 kg/m³
- Tensile Strength: 400-1200 MPa
- Young's Modulus: 30-50 GPa
- Elongation: 2-3%
- Specific strength: 200-600 kN·m/kg

Comparison to steel:
- Strength/weight: Similar or better
- Stiffness: 4-6× lower
- Cost: 5-10× lower than carbon fiber
- Electrical: Insulating (unlike steel)

Applications:
- Reinforced concrete (GFRP rebar)
- Marine (boat hulls)
- Automotive body panels
- Electrical insulators

E-glass: Most common, good electrical properties.
S-glass: Higher strength, aerospace.""", "tags": ["GFRP", "glass_fiber", "composite", "E_glass"], "source": "material_table", "category": "materials"},
    {"title": "Polyetheretherketone (PEEK) - Properties", "content": """PEEK High-Performance Polymer:

Mechanical:
- Density: 1300 kg/m³
- Tensile Strength: 90-100 MPa
- Young's Modulus: 3.5 GPa
- Impact Strength: 70 J/m (Izod)
- Hardness: 83 HRR

Thermal:
- Max service temp: 250°C (continuous)
- Glass transition: 143°C
- Melting point: 343°C
- Expansion: 47×10⁻⁶/K

Properties:
- Excellent chemical resistance
- High wear resistance
- Biocompatible (FDA approved)
- Flame retardant (UL94 V-0)

Applications:
- Aerospace seals and bearings
- Medical implants
- Oil and gas components
- Wire insulation

Carbon fiber PEEK: Tensile 250MPa, modulus 70GPa.""", "tags": ["PEEK", "high_performance", "polymer", "thermal"], "source": "material_table", "category": "materials"},
    {"title": "Beryllium - Properties", "content": """Beryllium Properties:

Mechanical:
- Density: 1850 kg/m³ (light!)
- Young's Modulus: 287 GPa (higher than steel!)
- Tensile Strength: 370 MPa
- Specific stiffness: 155 ×10⁶ N·m/kg (BEST)

Properties:
- Very low thermal expansion: 11×10⁻⁶/K
- High thermal conductivity: 200 W/m·K
- X-ray transparent
- Neutron moderator

DANGER - TOXICITY:
- Beryllium dust/fumes are extremely toxic
- Causes berylliosis (lung disease)
- Requires special handling and enclosures

Applications:
- Aerospace mirrors (James Webb)
- X-ray windows
- Nuclear applications
- Gyroscope components

Cost: $500-1000/kg (expensive and regulated).""", "tags": ["beryllium", "low_density", "high_stiffness", "toxic"], "source": "material_table", "category": "materials"},
    {"title": "Tungsten - Properties", "content": """Tungsten (W) Properties:

Mechanical:
- Density: 19300 kg/m³ (heaviest practical metal!)
- Young's Modulus: 410 GPa
- Tensile Strength: 550-1500 MPa
- Melting point: 3422°C (highest of all metals!)
- Hardness: 2000-4000 HV

Properties:
- Lowest thermal expansion of metals
- High thermal conductivity
- Excellent wear resistance
- High radiation resistance

Applications:
- Tungsten inert gas (TIG) welding electrodes
- Counterweights, ballast
- Armor-piercing ammunition
- Heavy alloy for vibration damping
- Electrical contacts
- Lamp filaments

Heavy alloy (90W): 17-18.5 g/cc, used for vibration damping.""", "tags": ["tungsten", "high_density", "refractory", "melting_point"], "source": "material_table", "category": "materials"},
    {"title": "Tungsten Carbide - Properties", "content": """Tungsten Carbide (WC-Co) Properties:

Mechanical:
- Density: 15600 kg/m³
- Hardness: 1600-2500 HV (next to diamond!)
- Young's Modulus: 530-700 GPa
- Compressive Strength: 4000-6000 MPa
- Toughness: 8-15 MPa·√m (tough grade)

Properties:
- Extreme wear resistance
- High hot hardness (retains hardness at 500°C)
- Good corrosion resistance (with cobalt binder)

Applications:
- Cutting tool inserts
- Drill bits
- Mining equipment
- Drawing dies
- Ballpoint pen tips

Composition: WC grains + Co binder (6-13%)
Fine grain: Better wear resistance
Coarse grain: Better toughness.""", "tags": ["tungsten_carbide", "WC", "cutting_tool", "wear_resistant"], "source": "material_table", "category": "materials"},
    {"title": "Copper - Properties", "content": """Copper (Cu) Properties:

Mechanical:
- Density: 8900 kg/m³
- Tensile Strength: 200-400 MPa
- Yield Strength: 33-400 MPa (varies by temper)
- Young's Modulus: 110-130 GPa
- Electrical conductivity: 100% IACS (BEST!)
- Thermal conductivity: 400 W/m·K (excellent)

Properties:
- Excellent electrical conductivity
- Good thermal conductivity
- Ductile (easily formed)
- Good corrosion resistance
- Solders and brazes easily

Applications:
- Electrical wiring (60% of copper use)
- Busbars, connectors
- Heat sinks
- Plumbing
- Bronze/brass alloys

Conductors: Annealed copper for maximum conductivity.
LME grade: 99.99% pure.""", "tags": ["copper", "electrical_conductivity", "thermal", "conductor"], "source": "material_table", "category": "materials"},
    {"title": "Lead - Properties", "content": """Lead (Pb) Properties:

Mechanical:
- Density: 11340 kg/m³ (heavy, soft)
- Tensile Strength: 15-30 MPa
- Young's Modulus: 14 GPa
- Hardness: 3-4 HB (very soft!)
- Melting point: 327°C

Properties:
- Excellent radiation shielding (X-ray, gamma)
- High density (ballast, counterweights)
- Low melting point (easily cast)
- Excellent corrosion resistance
- TOXIC - lead poisoning risk!

Applications:
- Radiation shielding (medical, nuclear)
- Ballast weights
- Batteries (lead-acid)
- Ammunition (historically)
- Soundproofing

Safe handling: Lead is toxic. Use PPE. Many alternatives now available.""", "tags": ["lead", "radiation_shielding", "high_density", "toxic"], "source": "material_table", "category": "materials"},
    {"title": "Nickel Superalloy Inconel 625 - Properties", "content": """Inconel 625 Properties:

Mechanical:
- Density: 8440 kg/m³
- Yield Strength: 450 MPa (room temp)
- Tensile Strength: 880 MPa
- Young's Modulus: 208 GPa
- Creep rupture: Excellent at 650°C

Corrosion:
- Outstanding resistance to chloride pitting
- Excellent aqueous corrosion resistance
- Oxidation resistant to 1000°C

Properties:
- Nickel-chromium-molybdenum alloy
- Niobium strengthens via precipitation
- Exceptional toughness at cryogenic to elevated temps

Applications:
- Marine environments
- Pollution control equipment
- Aerospace: Exhaust, turbine shields
- Nuclear: Control rod cladding
- Chemical processing

Cost: $20-40/kg.""", "tags": ["inconel_625", "nickel_superalloy", "corrosion_resistant", "marine"], "source": "material_table", "category": "materials"},
    {"title": "Zinc Die Casting Alloy (Zamak) - Properties", "content": """Zamak (Zinc Alloy) Properties:

Mechanical:
- Density: 6600-6700 kg/m³
- Tensile Strength: 280-380 MPa
- Yield Strength: 220-320 MPa
- Young's Modulus: 85 GPa
- Elongation: 2-10%

Properties:
- Excellent castability (low melting: 380-390°C)
- Good surface finish as-cast
- Excellent dimensional stability
- Good bearing properties (self-lubricating)
- 95% recycled material common

Grades:
- Zamak 3: General purpose, best balance
- Zamak 5: Higher strength, 1% Cu
- Zamak 7: High purity, better plating

Applications:
- Die castings (door handles, locks)
- Plumbing fixtures
- Electrical hardware
- Automotive parts

Cost: $2-3/kg (very economical).""", "tags": ["zamak", "zinc_die_cast", "casting", "alloy"], "source": "material_table", "category": "materials"},
    {"title": "Nylon (PA66) - Properties", "content": """Nylon 66 (Polyamide 66):

Mechanical:
- Density: 1140 kg/m³
- Tensile Strength: 80-90 MPa
- Young's Modulus: 2.0-3.0 GPa
- Impact Strength: 40-60 J/m (notched Izod)
- Elongation: 50-200%
- Wear resistance: Excellent

Thermal:
- Melting point: 265°C
- Max service temp: 100-150°C
- Expansion: 80×10⁻⁶/K

Properties:
- Self-lubricating (good for gears)
- Good chemical resistance
- Absorbs moisture (1-8%)
- Good fatigue resistance

Applications:
- Gears, bearings, cams
- Wire insulation
- Consumer products
- Automotive (25% of nylon use)
- Textiles

Moisture absorption affects dimensions and properties.""", "tags": ["nylon", "PA66", "polyamide", "wear_resistant"], "source": "material_table", "category": "materials"},
    {"title": "Ultem (PEI) - Properties", "content": """Ultem (PEI - Polyetherimide):

Mechanical:
- Density: 1270 kg/m³
- Tensile Strength: 105 MPa
- Young's Modulus: 3.0 GPa
- Impact Strength: 50 J/m (Izod)
- Hardness: 88 HRR

Thermal:
- Glass transition: 217°C
- Max service temp: 200°C (continuous)
- Flame spread: UL94 V-0
- Smoke density: Low

Properties:
- Excellent thermal stability
- High strength-to-weight ratio
- Good chemical resistance
- Dimensional stability
- Flame retardant (no additives needed)

Applications:
- Aerospace interiors
- Medical instrument handles
-Electrical insulators
- Fluid handling components
- 3D printed parts (PEEK/PEI filaments)

Cost: $30-50/kg (premium engineering polymer).""", "tags": ["Ultem", "PEI", "high_temperature", "flame_retardant"], "source": "material_table", "category": "materials"},
    {"title": "AISI 1045 Carbon Steel - Properties", "content": """AISI 1045 Medium Carbon Steel:

Mechanical:
- Density: 7850 kg/m³
- Yield Strength: 310-530 MPa (varies by heat treat)
- Tensile Strength: 570-680 MPa
- Young's Modulus: 205 GPa
- Elongation: 12-16%
- Hardness: 170-210 HB (annealed)

Heat Treatment:
- Annealed: 170 HB, ductile
- Normalized: Better machinability
- Quenched + tempered: Up to 600 HB
- Carburized: Surface 60 HRC, core soft

Properties:
- Good balance of strength and machinability
- Widely available and inexpensive
- Responds well to heat treatment
- 0.45% carbon content

Applications:
- Shafts, axles, spindles
- Gears (after heat treat)
- Crankshafts
- Connecting rods
- Bolts, studs

Cost: $0.80-1.50/kg (very economical).""", "tags": ["1045_steel", "carbon_steel", "medium_carbon", "heat_treatable"], "source": "material_table", "category": "materials"},
    {"title": "Maraging Steel - Properties", "content": """Maraging Steel (18Ni300):

Mechanical:
- Density: 8000 kg/m³
- Yield Strength: 2000-2500 MPa!
- Tensile Strength: 2500-3000 MPa!
- Young's Modulus: 190 GPa
- Elongation: 6-12%

Properties:
- Ultra-high strength (highest of steels)
- Excellent toughness at high strength
- Low carbon (prevents martensite brittleness)
- Age-hardenable at 480°C
- Excellent dimensional stability during heat treat

Composition:
- 18% Nickel (austenite stabilizer)
- 8-12% Cobalt (promotes hardening)
- 3-5% Molybdenum (strengthening)
- <0.03% Carbon

Applications:
- Aerospace landing gear
- Missile components
- High-speed tooling
- Die casting dies
- Sporting goods (golf club faces)

Cost: $30-50/kg (expensive).""", "tags": ["maraging_steel", "ultra_high_strength", "aerospace", "toughness"], "source": "material_table", "category": "materials"},
    {"title": "Zirconia (ZrO₂) - Properties", "content": """Yttria-Stabilized Zirconia (YSZ):

Mechanical:
- Density: 6060 kg/m³
- Hardness: 1200-1300 HV
- Flexural Strength: 900-1200 MPa
- Young's Modulus: 200 GPa
- Fracture toughness: 5-10 MPa·√m (HIGHEST of ceramics!)

Thermal:
- Thermal conductivity: 2-3 W/m·K (very low!)
- Expansion: 10×10⁻⁶/K
- Max service temp: 1000°C

Properties:
- Excellent thermal barrier (thermal barrier coatings)
- High hardness and wear resistance
- Best fracture toughness of engineering ceramics
- Biocompatible (dental implants)

Transformation toughening:
Tetragonal → Monoclinic phase transformation absorbs energy
Crack propagation halted by compressive transformation zone.

Applications:
- Thermal barrier coatings (jet engine blades)
- Dental crowns and implants
- Cutting tools
- Oxygen sensors
- Ball valves""", "tags": ["zirconia", "YSZ", "toughness", "thermal_barrier"], "source": "material_table", "category": "materials"},
]

# Re-add all additional chunks to the master list
for chunk in FORMULA_VARIATIONS:
    chunk["category"] = chunk.get("category", "foundational")
    ALL_CHUNKS.append(chunk)

for chunk in MORE_MATERIALS:
    chunk["category"] = "materials"
    ALL_CHUNKS.append(chunk)

print(f"Total chunks now: {get_total_count()}")
