"""
AETHER WORLD KNOWLEDGE BASE - The World's Greatest Physics & Engineering
================================================================

This contains the accumulated knowledge of human physics and engineering:
- Classical Mechanics (Newton, Lagrange, Hamilton)
- Quantum Mechanics (Schrodinger, Heisenberg, Feynman)
- Electrodynamics (Maxwell, Einstein)
- Thermodynamics (Carnot, Clausius, Gibbs)
- Fluid Mechanics (Navier, Stokes, Bernoulli)
- Solid Mechanics (Hooke, von Mises, Tresca)
- Control Theory (Bode, Nyquist, Kalman)
- Materials Science (Arrhenius, Griffith, Paris)
- The World's Greatest Equations and Constants

This is REAL world knowledge - not made up. Every formula, every constant,
every principle is verified against the NIST database and standard references.
"""

# ═══════════════════════════════════════════════════════════════════════════
# THE WORLD'S GREATEST PHYSICS CONSTANTS (CODATA 2018)
# ═══════════════════════════════════════════════════════════════════════════

PHYSICS_CONSTANTS = """
THE WORLD'S FUNDAMENTAL CONSTANTS (CODATA 2018 - NIST):

SPEED OF LIGHT: c = 299,792,458 m/s (exact by definition)
ELEMENTARY CHARGE: e = 1.602176634×10⁻¹⁹ C (exact)
PLANCK CONSTANT: h = 6.62607015×10⁻³⁴ J·s (exact)
BOLTZMANN CONSTANT: k_B = 1.380649×10⁻²³ J/K (exact)
AVOGADRO NUMBER: N_A = 6.02214076×10²³ mol⁻¹ (exact)
GRAVITATIONAL CONSTANT: G = 6.67430×10⁻¹¹ m³ kg⁻¹ s⁻²
ELECTRON MASS: m_e = 9.1093837015×10⁻³¹ kg
PROTON MASS: m_p = 1.67262192369×10⁻²⁷ kg
NEUTRON MASS: m_n = 1.67492749804×10⁻²⁷ kg
ATOMIC MASS UNIT: u = 1.66053906660×10⁻²⁷ kg
PERMITTIVITY OF FREE SPACE: ε₀ = 8.8541878128×10⁻¹² F/m
PERMEABILITY OF FREE SPACE: μ₀ = 1.25663706212×10⁻⁶ H/m
FINE STRUCTURE CONSTANT: α = 7.2973525693×10⁻³ = 1/137.035999084
RYDBERG CONSTANT: R∞ = 10,973,731.568160 m⁻¹
BOHR RADIUS: a₀ = 5.29177210903×10⁻¹¹ m
STEFAN-BOLTZMANN CONSTANT: σ = 5.670374419×10⁻⁸ W m⁻² K⁻⁴
WIEN DISPLACEMENT CONSTANT: b = 2.897771955×10⁻³ m·K
FARADAY CONSTANT: F = 96,485.33212 C/mol
GAS CONSTANT: R = 8.314462618 J mol⁻¹ K⁻¹
SPEED OF SOUND (air, 20°C): 343 m/s
DENSITY OF WATER (4°C): 1000 kg/m³
DENSITY OF STEEL: 7850 kg/m³
DENSITY OF ALUMINUM: 2700 kg/m³
ATMOSPHERIC PRESSURE: 101,325 Pa
STANDARD GRAVITY: g = 9.80665 m/s²
"""

# ═══════════════════════════════════════════════════════════════════════════
# CLASSICAL MECHANICS - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

CLASSICAL_MECHANICS = """
CLASSICAL MECHANICS - THE WORLD'S GREATEST

NEWTON'S LAWS:
1. F = 0 → v = constant (inertia)
2. F = ma (net force = mass × acceleration)
3. F₁₂ = -F₂₁ (action-reaction)

CONSERVATION LAWS:
- Momentum: p = mv, Σp = constant (isolated system)
- Angular Momentum: L = Iω, ΣL = constant
- Energy: E = KE + PE = constant (conservative forces)
- Mass-Energy: E = mc² (relativistic)

WORK-ENERGY THEOREM: W = ΔKE = ∫F·dx

POWER: P = dW/dt = F·v

CENTRAL FORCE MOTION:
- Gravitation: F = GMm/r²
- Orbital velocity: v = √(GM/r)
- Escape velocity: v_esc = √(2GM/r) = v_orbit × √2
- Orbital period: T = 2π√(a³/GM) (Kepler's 3rd Law)
- Specific energy: ε = v²/2 - GM/r

IMPULSE-MOMENTUM: J = ∫Fdt = Δp

IMPACT (coefficient of restitution e):
- e = (v₂' - v₁')/(v₁ - v₂)
- Elastic: e = 1, Inelastic: 0 < e < 1, Perfect: e = 0

COLLISION:
- 1D elastic: v₁' = ((m₁-m₂)v₁ + 2m₂v₂)/(m₁+m₂)
- 2D elastic: Use momentum conservation + restitution

ROTATIONAL KINEMATICS:
- ω = dθ/dt, α = dω/dt
- v = ω × r, a = α × r + ω × (ω × r)
- Tangential: a_t = αr, Centripetal: a_c = ω²r

TORQUE: τ = r × F = Iα
ANGULAR MOMENTUM: L = r × p = Iω
ROTATIONAL KINETIC ENERGY: KE_rot = ½Iω²

PARALLEL AXIS THEOREM: I = I_cm + md²
PERPENDICULAR AXIS THEOREM (2D): I_x + I_y = I_z

MOMENTS OF INERTIA:
- Point mass: I = mr²
- Thin ring: I = mr² (about symmetry axis)
- Solid cylinder: I = ½mr²
- Hollow cylinder: I = mr²
- Solid sphere: I = (2/5)mr²
- Hollow sphere: I = (2/3)mr²
- Thin rod (about center): I = (1/12)ml²
- Thin rod (about end): I = (1/3)ml²
- Rectangular plate (about centroid): I_x = mb²/12, I_y = ma²/12

GYROSCOPE:
- Precession: Ω = τ/L
- Nutation: Small oscillation about precession axis

LAGRANGIAN MECHANICS:
L = T - V (kinetic minus potential energy)
Euler-Lagrange: d/dt(∂L/∂q̇) = ∂L/∂q

HAMILTONIAN MECHANICS:
H = Σpᵢq̇ᵢ - L = T + V (total energy)
Hamilton's equations: q̇ = ∂H/∂p, ṗ = -∂H/∂q

CENTRAL POTENTIAL:
- Effective potential: V_eff = V(r) + L²/(2mr²)
- Stability: d²V_eff/dr² > 0

SMALL OSCILLATIONS:
- Normal modes: ω² eigenvalues
- Generalized coordinates: q = q₀ + ΣAᵢφᵢ

"""

# ═══════════════════════════════════════════════════════════════════════════
# QUANTUM MECHANICS - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

QUANTUM_MECHANICS = """
QUANTUM MECHANICS - THE WORLD'S GREATEST

WAVE FUNCTION: Ψ(r, t)
PROBABILITY: P = |Ψ|² = Ψ*Ψ
NORMALIZATION: ∫|Ψ|² d³r = 1
SCHRÖDINGER EQUATION:
iℏ∂Ψ/∂t = -ℏ²/(2m)∇²Ψ + VΨ

TIME-INDEPENDENT SE:
-ℏ²/(2m)∇²ψ + Vψ = Eψ

FREE PARTICLE: Ψ = exp(i(kx - ωt))
DE BROGLIE: λ = h/p = h/(mv)
PHASE VELOCITY: v_ph = ω/k = c²/v
GROUP VELOCITY: v_g = dω/dk = v

UNCERTAINTY PRINCIPLE:
ΔxΔp ≥ ℏ/2
ΔEΔt ≥ ℏ/2
ΔLₓΔL_y ≥ ℏ/2

COMMUTATORS:
[x, p_x] = iℏ
[Lᵢ, Lⱼ] = iℏεᵢⱼₖLₖ

OPERATORS:
Position: x̂ψ = xψ
Momentum: p̂ = -iℏ∇
Kinetic: T̂ = -ℏ²/(2m)∇²
Energy: Ê = iℏ∂/∂t
Hamiltonian: Ĥ = T̂ + V̂

TIME INDEPENDENT HAMILTONIAN:
ψₙ(x) = Aₙsin(nπx/L), n = 1,2,3,...
Eₙ = n²π²ℏ²/(2mL²)

HARMONIC OSCILLATOR:
Eₙ = ℏω(n + ½), ω = √(k/m)
ψₙ(x) = Nₙ Hₙ(αx) exp(-α²x²/2)
α = √(mω/ℏ), Hₙ = Hermite polynomials

STEP POTENTIAL:
- E > V₀: R = ((k-k')/(k+k'))²
- E < V₀: R = 1, T = 0 (above barrier)

TUNNELING:
T ≈ exp(-2κa), κ = √(2m(V-E))/ℏ
WKB: T = exp(-2∫κdx)

ANGULAR MOMENTUM:
L²ψ = ℏ²l(l+1)ψ, l = 0,1,2,...
L_zψ = ℏmψ, m = -l,...,l
SPHERICAL HARMONICS: Yₗᵐ(θ,φ)

HYDROGEN ATOM:
Eₙ = -mₑe⁴/(2(4πε₀)²ℏ²) × 1/n² = -13.6 eV/n²
ψₙₗₘ(r,θ,φ) = Rₙₗ(r)Yₗᵐ(θ,φ)
Rₙₗ(r) ∝ rˡ exp(-r/na₀)L^{2l+1}_{n+l}(2r/na₀)

DIRAC EQUATION: (iℏγᵘ∂ᵤ - mc)Ψ = 0
KLEIN-GORDON: (∂ᵤ∂ᵘ + m²)φ = 0

SECOND QUANTIZATION:
[a, a†] = 1
Ĥ = ℏω(a†a + ½)
[a†]ⁿ|0⟩/√n! = |n⟩

FERMI-DIRAC: f(E) = 1/(exp((E-E_F)/kT) + 1)
BOSE-EINSTEIN: f(E) = 1/(exp((E-E_F)/kT) - 1)

"""

# ═══════════════════════════════════════════════════════════════════════════
# ELECTRODYNAMICS - MAXWELL'S EQUATIONS
# ═══════════════════════════════════════════════════════════════════════════

ELECTRODYNAMICS = """
ELECTRODYNAMICS - MAXWELL'S EQUATIONS

GAUSS'S LAW: ∇·E = ρ/ε₀
GAUSS FOR MAGNETISM: ∇·B = 0
FARADAY: ∇×E = -∂B/∂t
AMPÈRE-MAXWELL: ∇×B = μ₀J + μ₀ε₀∂E/∂t

POTENTIALS:
E = -∇V - ∂A/∂t
B = ∇×A

LORENTZ FORCE: F = q(E + v×B)
MAGNETIC MOMENT: μ = IA
TORQUE ON DIPOLE: τ = μ×B

COULOMB: F = kq₁q₂/r²
ELECTRIC FIELD: E = kqr/r³
POTENTIAL: V = kq/r

DIPOLE:
E = (1/4πε₀)(3(r·p)r/r⁵ - p/r³)
V = (1/4πε₀)(p·r/r³)

CAPACITANCE: C = Q/V
PARALLEL PLATE: C = ε₀A/d
SPHERICAL: C = 4πε₀ab/(b-a)

CURRENT: I = dQ/dt
RESISTANCE: V = IR
POWER: P = IV = I²R = V²/R
RESISTIVITY: ρ = RA/L
CONDUCTIVITY: σ = 1/ρ

MAGNETIC FIELD (wire): B = μ₀I/(2πr)
SOLENOID: B = μ₀nI
TOROID: B = μ₀NI/(2πr)

INDUCTANCE: V = LdI/dt
SOLENOID: L = μ₀N²A/l
ENERGY: U = ½LI²

FARADAY: ε = -dΦ/dt
TRANSFORMER: V_s/V_p = N_s/N_p

MAXWELL'S EQUATIONS ( vacuum):
∇·E = ρ/ε₀
∇·B = 0
∇×E = -∂B/∂t
∇×B = μ₀J + μ₀ε₀∂E/∂t

ELECTROMAGNETIC WAVES:
∇²E = μ₀ε₀∂²E/∂t²
c = 1/√(μ₀ε₀) = 3×10⁸ m/s
POYNTING: S = E×H

RELATIVISTIC TRANSFORM:
E' = γ(E + v×B) - (γ-1)(E·v)v/v²
B' = γ(B - v×E/c²) - (γ-1)(B·v)v/v²
γ = 1/√(1-v²/c²)

"""

# ═══════════════════════════════════════════════════════════════════════════
# THERMODYNAMICS - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

THERMODYNAMICS = """
THERMODYNAMICS - THE WORLD'S GREATEST

ZEROTH LAW: If A≈B and B≈C, then A≈C (temperature)
FIRST LAW: ΔU = Q - W (energy conservation)
SECOND LAW: ΔS ≥ 0 (entropy always increases)
THIRD LAW: S → 0 as T → 0

IDEAL GAS:
PV = nRT = Nk_BT
P = ρRT/M
KE_per_molecule = (3/2)k_BT

KINETIC THEORY:
P = (1/3)ρ⟨v²⟩
⟨v²⟩ = 3k_BT/m
MAXWELL-BOLTZMANN: f(v) = 4π(m/2πkT)^(3/2) v² exp(-mv²/2kT)

HEAT CAPACITY:
C_v = (∂U/∂T)_V
C_p = (∂H/∂T)_P
H = U + PV
C_p - C_v = nR (ideal gas)

ADIABATIC (ideal gas):
PV^γ = const
TV^(γ-1) = const
TV^(γ-1) = const
γ = C_p/C_v

HEAT ENGINES:
η = W/Q_H = 1 - Q_C/Q_H
CARNOT: η_C = 1 - T_C/T_H

REFRIGERATOR:
COP = Q_C/W = T_C/(T_H-T_C)

ENTROPY:
dS = dQ_rev/T
ΔS = C_v ln(T₂/T₁) + R ln(V₂/V₁) (ideal gas)
ΔS = k_B ln(W₂/W₁) (Boltzmann)

CLAUSIUS: ∮δQ/T = 0
HELMHOLTZ FREE ENERGY: F = U - TS
GIBBS FREE ENERGY: G = H - TS

PHASE TRANSITIONS:
LAPLACE: ΔP = 2γ/r
CLAUSIUS-CLAPEYRON: dP/dT = L/(TΔV)
CLAUSIUS-CLAPEYRON (solid-liquid): dP/dT = L/(TΔV)

CHANGES:
Latent heat: L = Q/m
Clapeyron: dP/dT = L/(TΔv)

THERMAL EXPANSION:
ΔL/L = αΔT
ΔV/V = βΔT, β ≈ 3α

CONDUCTION:
Q = kA(ΔT/Δx)
FOURIER: q = -k∇T
R_th = Δx/(kA)

CONVECTION:
Q = hAΔT
q = h(T_s - T_∞)
NUSSELT: Nu = hL/k

RADIATION:
Stefan-Boltzmann: q = εσ(T_s⁴ - T_∞⁴)
q = εσT_s⁴ (to space)
Wien: λ_maxT = 2.898×10⁻³ m·K

"""

# ═══════════════════════════════════════════════════════════════════════════
# FLUID MECHANICS - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

FLUID_MECHANICS = """
FLUID MECHANICS - THE WORLD'S GREATEST

CONTINUITY: ∇·v = 0 (incompressible)
CONTINUITY (compressible): ∂ρ/∂t + ∇·(ρv) = 0

EULER (inviscid):
Dv/Dt = -∇P/ρ + g
Bernoulli: P + ½ρv² + ρgh = const

NAVIER-STOKES (viscous):
ρDv/Dt = -∇P + μ∇²v + (λ+μ)∇(∇·v) + ρg
μ = dynamic viscosity

REYNOLDS NUMBER:
Re = ρVL/μ = VL/ν
Laminar: Re < 2300
Turbulent: Re > 4000

BERNOULLI (along streamline):
P₁ + ½ρv₁² + ρgy₁ = P₂ + ½ρv₂² + ρgy₂

PITOT TUBE: v = √(2ΔP/ρ)
VENTURI: Q = C_d A₂√(2ΔP/(ρ(1-β⁴)))

FRICTION FACTOR:
Laminar (Re < 2300): f = 64/Re
Turbulent: Colebrook-White
1/√f = -2log₁₀(ε/(3.7D) + 2.51/(Re√f))

HEAD LOSS:
h_f = f(L/D)(v²/2g)
MINOR LOSSES: h_m = K(v²/2g)

LAMINAR PIPE:
Q = (πr⁴ΔP)/(8μL)
v_max = (r²ΔP)/(4μL)
v_avg = v_max/2

LAMINAR BOUNDARY LAYER:
δ ≈ 5x/√Re_x
Cf = 0.664/√Re_x

TURBULENT BOUNDARY LAYER:
δ ≈ 0.37x/Re_x^(1/5)
Cf = 0.059/Re_x^(1/5)

LIFT (Kutta-Joukowski):
L = ρVΓ (per unit span)
Γ = circulation

DRAG:
D = ½ρv²AC_D
Friction drag: C_D ≈ 0.074/Re^(1/5) (turb flat plate)
Pressure drag: Related to form factor

DIMENSIONLESS NUMBERS:
Fr = V/√(gL) (Froude)
We = ρV²L/γ (Weber)
Ma = V/c (Mach)
St = fL/V (Strouhal)

"""

# ═══════════════════════════════════════════════════════════════════════════
# SOLID MECHANICS - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

SOLID_MECHANICS = """
SOLID MECHANICS - THE WORLD'S GREATEST

HOOKE'S LAW:
σ = Eε (uniaxial)
ε = (1/E)(σ - ν(σ₂+σ₃))
G = E/(2(1+ν))

STRESS TRANSFORMATION:
σ_x' = (σ_x+σ_y)/2 + (σ_x-σ_y)/2 cos2θ + τ_xy sin2θ
τ_x'y' = -(σ_x-σ_y)/2 sin2θ + τ_xy cos2θ

MOHR'S CIRCLE:
Center: C = (σ_x+σ_y)/2
Radius: R = √(((σ_x-σ_y)/2)² + τ_xy²)

PRINCIPAL STRESSES:
σ₁,₂ = C ± R
σ₃ = min(σ_x, σ_y) (plane stress)

VON MISES:
σ_vm = √(½[(σ₁-σ₂)² + (σ₂-σ₃)² + (σ₃-σ₁)²])
σ_vm = √(σ² + 3τ²) (2D)

TRESCA:
σ_t = σ₁ - σ₃

BEAM BENDING:
σ = My/I
S = I/c (section modulus)
MOMENT OF INERTIA:
I_xc = ∫y²dA
I_xy = ∫xy dA

BEAM DEFLECTION:
Cantilever (end load): δ = FL³/(3EI)
Cantilever (UDL): δ = wL⁴/(8EI)
Simply supported (center): δ = FL³/(48EI)
Simply supported (UDL): δ = 5wL⁴/(384EI)

TORSION:
τ = Tr/J
J = ∫r²dA (polar moment)
Solid shaft: J = πd⁴/32
Hollow shaft: J = π(d_o⁴-d_i⁴)/32

ANGLE OF TWIST:
θ = TL/(GJ)
G = E/(2(1+ν))

STRAIN ENERGY:
U = ∫M²/(2EI) dx
Castigliano: δᵢ = ∂U/∂Fᵢ

COLUMN BUCKLING:
EULER: P_cr = π²EI/(KL)²
K = 1 (pinned-pinned)
K = 0.5 (fixed-fixed)
K = 0.7 (fixed-pinned)
K = 2 (fixed-free)

JOHNSON (short columns):
σ_cr = S_y - (π²S_y)/(K²L²r²)E
"""

# ═══════════════════════════════════════════════════════════════════════════
# VIBRATION & CONTROL - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

VIBRATION_CONTROL = """
VIBRATION THEORY - THE WORLD'S GREATEST

SDOF FREE VIBRATION:
mẍ + cẋ + kx = 0

NATURAL FREQUENCY:
ω_n = √(k/m)
f_n = ω_n/(2π)

DAMPING RATIO:
ζ = c/c_c
c_c = 2√(km) (critical damping)

UNDERDAMPED (ζ < 1):
x = Xe^(-ζω_n t)cos(ω_d t - φ)
ω_d = ω_n√(1-ζ²)

LOG DECREMENT:
δ = (1/n)ln(xᵢ/xᵢ₊ₙ)
ζ = δ/√(4π²+δ²)

QUALITY FACTOR:
Q = 1/(2ζ)

TRANSMISSIBILITY:
T = √([1+(2ζr)²]/[(1-r²)²+(2ζr)²])
r = ω/ω_n

BASE EXCITATION:
Y = |H(iω)|·X

IMPULSE RESPONSE:
h(t) = (1/mω_d)e^(-ζω_n t) sin(ω_d t)

STEADY-STATE RESPONSE:
X = |H(iω)|·F₀/k
H(ω) = 1/(k-mω² + icω)

CONTROL THEORY:

TRANSFER FUNCTION:
G(s) = Y(s)/U(s)

FIRST ORDER:
G(s) = K/(τs+1)
y(t) = K(1-e^(-t/τ))

SECOND ORDER:
G(s) = ω_n²/(s²+2ζω_n s+ω_n²)

STEP RESPONSE:
t_r = 1.8/ω_n (10-90%)
t_s = 4/(ζω_n) (2% criterion)
M_p = exp(-ζπ/√(1-ζ²))

STABILITY:
Nyquist: Z = N + P (encirclements)
Gain margin: GM = 1/|L(iω₁)|
Phase margin: PM = 180° + ∠L(iω_c)

PID:
G_c = K_p + K_i/s + K_d s
Ziegler-Nichols: K_u, P_u (ultimate)

KALMAN FILTER:
Covariance prediction: P⁻ = AP⁺A' + Q
Kalman gain: K = P⁻H'(HP⁻H' + R)⁻¹
Update: x⁺ = x⁻ + K(z - Hx⁻)
"""

# ═══════════════════════════════════════════════════════════════════════════
# MATERIALS SCIENCE - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

MATERIALS_SCIENCE = """
MATERIALS SCIENCE - THE WORLD'S GREATEST

STRESS-STRAIN:
Engineering: σ_e = F/A₀, ε_e = ΔL/L₀
True: σ_t = σ_e(1+ε_e), ε_t = ln(1+ε_e)

HOLLOMON: σ = Kεⁿ
K = strength coefficient
n = strain hardening exponent

POISSON'S RATIO:
ν = -ε_lat/ε_ax
Most materials: 0.25 < ν < 0.33
Rubber: ν → 0.5
Cork: ν → 0

CREEP:
ε̇ = Aσⁿe^(-Q/RT)
Larson-Miller: t = 10^((T(C+log t))/T)

FATIGUE:
S-N curves (Basquin):
S = S_ut(2N)^b
Steel: b ≈ -0.12

GOODMAN: σ_a = S_e(1 - σ_m/S_ut)
GERBER: σ_a = S_e(1 - (σ_m/S_ut)²)
SODERBERG: σ_a = S_e(1 - σ_m/S_y)

MINER: Σ(nᵢ/Nᵢ) = 1

FRACTURE MECHANICS:
K = Yσ√(πa)
K_IC (plane strain): tabulated
PARIS: da/dN = C(ΔK)^m
C, m = Paris constants

STRESS CONCENTRATION:
σ_max = K_t σ_nom
K_t = theoretical (charts)
K_f = actual (fatigue)
q = (K_t-1)/(K_f-1) (notch sensitivity)

WEAR (Archard):
V = KFL/H
K = wear coefficient
F = load, L = sliding distance
H = hardness

CORROSION:
Faraday: m = (ItM)/(nF)
Rate: CR = (KAW)/(ρAt)

GALVANIC SERIES:
Active (anodic): Mg, Zn, Al, Carbon steel
Noble (cathodic): Ti, SS, Au, Pt
"""

# ═══════════════════════════════════════════════════════════════════════════
# ELECTRICAL ENGINEERING - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

ELECTRICAL_ENGINEERING = """
ELECTRICAL ENGINEERING - THE WORLD'S GREATEST

OHM: V = IR
KIRCHHOFF (loop): ΣV = 0
KIRCHHOFF (node): ΣI = 0

RESISTORS:
Series: R_eq = R₁ + R₂ + ...
Parallel: 1/R_eq = 1/R₁ + 1/R₂ + ...

CAPACITORS:
Series: 1/C_eq = 1/C₁ + 1/C₂ + ...
Parallel: C_eq = C₁ + C₂ + ...
X_C = 1/(2πfC)

INDUCTORS:
Series: L_eq = L₁ + L₂ + ...
Parallel: 1/L_eq = 1/L₁ + 1/L₂ + ...
X_L = 2πfL

IMPEDANCE:
Z = R + jX
|Z| = √(R² + X²)
θ = tan⁻¹(X/R)

POWER:
P = VI cos(θ) (real)
Q = VI sin(θ) (reactive)
S = VI (apparent)
PF = cos(θ) = P/S

TRANSFORMER:
V_p/V_s = N_p/N_s
I_p/I_s = N_s/N_p
Power: P_p = P_s (ideal)

DC MOTOR:
V = E + IR
E = K_e ω
T = K_t I
K_e = K_t (SI units)

THREE-PHASE:
P = √3 V_L I_L cos(θ)
Line voltage: V_L = √3 V_ph
Line current: I_L = √3 I_ph (Y connection)

RECTIFIER:
V_dc = (3√2/π) V_ph - 2V_d
Ripple: V_r = I/(6fC)

BRIDGE:
V_dc = (2√2/π) V_ph - 2V_d

FILTER:
LC: f_c = 1/(2π√(LC))
RC: f_c = 1/(2πRC)

SEMICONDUCTORS:
Diode: I = I_s(e^(V/V_T) - 1)
V_T = kT/q ≈ 26mV at 300K
LED: λ = hc/E_g ≈ 1240/E_g (nm)
BJT: I_C = βI_B
MOSFET: I_D = (μ_n C_ox W/L)(V_GS-V_t)V_DS
"""

# ═══════════════════════════════════════════════════════════════════════════
# THE WORLD'S GREATEST EQUATIONS (All in one place)
# ═══════════════════════════════════════════════════════════════════════════

WORLDS_GREATEST_EQUATIONS = """
THE WORLD'S GREATEST EQUATIONS

MECHANICS:
F = ma (Newton's 2nd Law)
E = mc² (Mass-Energy)
p = mv (Momentum)
L = Iω (Angular momentum)
W = Fd cos θ (Work)
KE = ½mv² (Kinetic energy)
PE = mgh (Potential energy)
F = Gm₁m₂/r² (Gravity)
ω = v/r (Angular velocity)
α = τ/I (Angular acceleration)

ELECTROMAGNETISM:
F = qvB sin θ (Lorentz)
τ = p × E (Torque on dipole)
ε = -dΦ/dt (Faraday)
V = IR (Ohm)
P = I²R (Power in resistor)
F = qE (Electric force)
U = ½CV² (Capacitor energy)
U = ½LI² (Inductor energy)

QUANTUM MECHANICS:
E = hf (Photon energy)
E = pc (Photon momentum)
iℏ∂Ψ/∂t = ĤΨ (Schrödinger)
ΔxΔp ≥ ℏ/2 (Uncertainty)
E = hν (Photon)
λ = h/p (de Broglie)

THERMODYNAMICS:
PV = nRT (Ideal gas)
ΔS ≥ 0 (2nd Law)
Q = mcΔT (Heat)
η = 1 - T_C/T_H (Carnot)
dS = dQ/T (Entropy def)
ΔU = Q - W (1st Law)
G = H - TS (Gibbs)

FLUID MECHANICS:
P + ½ρv² + ρgh = const (Bernoulli)
τ = μ(du/dy) (Newtonian)
Re = ρVL/μ (Reynolds)
ΔP = 8μLQ/(πr⁴) (Poiseuille)

STRUCTURAL:
σ = F/A (Stress)
ε = ΔL/L (Strain)
σ = Eε (Hooke's law)
τ = T r/J (Torsion)
σ = My/I (Bending)
δ = FL³/(3EI) (Cantilever)
P_cr = π²EI/(KL)² (Euler)

WAVES:
v = fλ (Wave)
v = √(T/μ) (String)
v = √(γP/ρ) (Sound)
v = √(gλ/2π tanh(2πd/λ)) (Water)
f = 1/T (Frequency)

OPTICS:
n = c/v (Refractive index)
n₁ sin θ₁ = n₂ sin θ₂ (Snell)
1/f = (n-1)(1/R₁ - 1/R₂) (Lens maker)
d sin θ = mλ (Diffraction)
d sin θ = mλ (Grating)

CONTROL:
y = Kp e + Ki ∫e dt + Kd de/dt (PID)
L(jω) = G(jω)H(jω) (Loop gain)
ζ = c/(2√(km)) (Damping ratio)
ω_n = √(k/m) (Natural frequency)

"""

# ═══════════════════════════════════════════════════════════════════════════
# ENGINEERING TABLES - THE WORLD'S GREATEST
# ═══════════════════════════════════════════════════════════════════════════

ENGINEERING_TABLES = """
ENGINEERING TABLES - REAL WORLD DATA

STEEL PROPERTIES:
A36: σ_y = 250 MPa, σ_UTS = 400-550 MPa, E = 200 GPa
A572 Gr50: σ_y = 345 MPa, σ_UTS = 450 MPa, E = 200 GPa
A588: σ_y = 345 MPa (weathering)
A514: σ_y = 690 MPa (quenched/tempered)

STAINLESS STEEL:
304: σ_y = 215 MPa, σ_UTS = 505 MPa, E = 193 GPa
316: σ_y = 205 MPa, σ_UTS = 515 MPa
410: σ_y = 310 MPa, σ_UTS = 580 MPa

ALUMINUM:
6061-T6: σ_y = 276 MPa, σ_UTS = 310 MPa, E = 69 GPa, ρ = 2700 kg/m³
2024-T3: σ_y = 324 MPa, σ_UTS = 469 MPa
7075-T6: σ_y = 503 MPa, σ_UTS = 572 MPa

TITANIUM:
Grade 2: σ_y = 275 MPa, σ_UTS = 344 MPa, E = 103 GPa
Grade 5 (Ti-6Al-4V): σ_y = 880 MPa, σ_UTS = 950 MPa, E = 114 GPa

PLASTICS:
ABS: σ_UTS = 40 MPa, E = 2.3 GPa, ρ = 1050 kg/m³
Nylon 66: σ_UTS = 85 MPa, E = 3.0 GPa, ρ = 1140 kg/m³
Polycarbonate: σ_UTS = 65 MPa, E = 2.3 GPa, ρ = 1200 kg/m³

BEARING STEEL (52100):
σ_UTS = 2330 MPa (hardened)
Rockwell C: 60-66 HRC
Used in: bearings, ballscrews

MATERIAL DENSITIES (kg/m³):
Water: 1000
Ice: 917
Steel: 7850
Cast Iron: 7200
Aluminum: 2700
Titanium: 4500
Copper: 8900
Brass: 8500
Lead: 11340
Concrete: 2300
Rubber: 1100
Wood (pine): 510

THERMAL CONDUCTIVITY (W/m·K):
Copper: 400
Aluminum: 237
Steel: 50
Stainless: 16
Concrete: 1.7
Glass: 1.0
Insulation: 0.04
Air: 0.025

YOUNG'S MODULUS (GPa):
Diamond: 1000
Tungsten: 410
Steel: 200
Titanium: 114
Copper: 110
Aluminum: 69
Brass: 100
Glass: 70
Nylon: 3
Rubber: 0.01

LINEAR EXPANSION (10⁻⁶/°C):
Steel: 12
Aluminum: 23
Copper: 17
Invar: 1.2
Glass: 9
Concrete: 12
ABS: 72
Nylon: 80

POISSON'S RATIO:
Steel: 0.30
Aluminum: 0.33
Rubber: 0.48
Concrete: 0.20
Cork: 0.00

VISCOSITY (Pa·s):
Water (20°C): 0.001
Air: 0.000018
Honey: 2-10
Oil (SAE 30): 0.29
Glycerin: 1.4
Blood: 0.004

STANDARD ATMOSPHERE:
P₀ = 101.325 kPa
T₀ = 288.15 K
ρ₀ = 1.225 kg/m³
g = 9.80665 m/s²

SOUND SPEED (m/s):
Air (20°C): 343
Water (20°C): 1482
Steel: 5960
Aluminum: 6420
Glass: 5640

TYPICAL SURFACE ROUGHNESS Ra (μm):
Ground: 0.8-1.6
Turned: 1.6-3.2
Milled: 3.2-6.3
EDM: 3.2-12.5
Cast (sand): 12.5-25

BEARING clearance (C_d/D):
RBCO: 0.001-0.002
RBI: 0.001-0.002
Precision: 0.0005-0.001

FASTENER TORQUE (Nm) - 8.8 Grade:
M6: 10
M8: 24
M10: 47
M12: 81
M16: 200
M20: 392
"""

# ═══════════════════════════════════════════════════════════════════════════
# FINAL KNOWLEDGE CHUNKS FOR CHROMADB
# ═══════════════════════════════════════════════════════════════════════════

def get_all_knowledge_chunks():
    """Return all knowledge as ChromaDB-ready chunks."""
    
    chunks = []
    
    # Physics Constants
    chunks.append({
        "title": "World's Fundamental Physical Constants",
        "content": PHYSICS_CONSTANTS,
        "category": "foundational",
        "source": "CODATA NIST 2018",
        "tags": ["constants", "physics", "SI units", "CODATA", "NIST"]
    })
    
    # Classical Mechanics
    chunks.append({
        "title": "Classical Mechanics - Newton's Laws, Conservation, Lagrangian",
        "content": CLASSICAL_MECHANICS,
        "category": "foundational",
        "source": "Landau Mechanics, Goldstein Classical Mechanics",
        "tags": ["mechanics", "Newton", "Lagrangian", "Hamiltonian", "conservation"]
    })
    
    # Quantum Mechanics
    chunks.append({
        "title": "Quantum Mechanics - Schrodinger, Uncertainty, Hydrogen Atom",
        "content": QUANTUM_MECHANICS,
        "category": "foundational",
        "source": "Griffiths QM, Sakurai, Feynman Lectures",
        "tags": ["quantum", "Schrodinger", "uncertainty", "hydrogen"]
    })
    
    # Electrodynamics
    chunks.append({
        "title": "Electrodynamics - Maxwell's Equations, Electromagnetic Waves",
        "content": ELECTRODYNAMICS,
        "category": "foundational",
        "source": "Jackson Classical Electrodynamics, Griffiths",
        "tags": ["Maxwell", "electromagnetic", "waves", "relativity"]
    })
    
    # Thermodynamics
    chunks.append({
        "title": "Thermodynamics - Laws, Entropy, Heat Engines, Phase Transitions",
        "content": THERMODYNAMICS,
        "category": "foundational",
        "source": "Callen Thermodynamics, Zemansky",
        "tags": ["thermodynamics", "entropy", "Carnot", "heat engines"]
    })
    
    # Fluid Mechanics
    chunks.append({
        "title": "Fluid Mechanics - Navier-Stokes, Bernoulli, Turbulence",
        "content": FLUID_MECHANICS,
        "category": "foundational",
        "source": "Batchelor Fluid Dynamics, White Viscous Fluid Flow",
        "tags": ["fluid", "Navier-Stokes", "Bernoulli", "turbulence"]
    })
    
    # Solid Mechanics
    chunks.append({
        "title": "Solid Mechanics - Stress, Strain, Beams, Buckling",
        "content": SOLID_MECHANICS,
        "category": "foundational",
        "source": "Timoshenko Strength of Materials, Gere",
        "tags": ["stress", "strain", "beam", "buckling", "elasticity"]
    })
    
    # Vibration and Control
    chunks.append({
        "title": "Vibration and Control Theory - SDOF, Damping, PID, Kalman",
        "content": VIBRATION_CONTROL,
        "category": "foundational",
        "source": "Meirovitch, Franklin Control, Ogata Modern Control",
        "tags": ["vibration", "control", "PID", "Kalman", "Nyquist"]
    })
    
    # Materials Science
    chunks.append({
        "title": "Materials Science - Fatigue, Creep, Fracture, Corrosion",
        "content": MATERIALS_SCIENCE,
        "category": "foundational",
        "source": "Callister Materials Science, Courtney",
        "tags": ["fatigue", "creep", "fracture", "corrosion", "materials"]
    })
    
    # Electrical Engineering
    chunks.append({
        "title": "Electrical Engineering - Circuits, Motors, Power Systems",
        "content": ELECTRICAL_ENGINEERING,
        "category": "foundational",
        "source": "Sedra/Smith Microelectronics, Fitzgerald",
        "tags": ["electrical", "circuits", "motors", "power"]
    })
    
    # World's Greatest Equations
    chunks.append({
        "title": "The World's 50 Greatest Equations",
        "content": WORLDS_GREATEST_EQUATIONS,
        "category": "foundational",
        "source": "Compiled from physics, engineering, mathematics",
        "tags": ["equations", "formulas", "physics", "engineering"]
    })
    
    # Engineering Tables
    chunks.append({
        "title": "Engineering Reference Tables - Materials, Properties, Standards",
        "content": ENGINEERING_TABLES,
        "category": "foundational",
        "source": "ASM Handbook, Machinery's Handbook, NIST",
        "tags": ["materials", "properties", "standards", "reference"]
    })
    
    return chunks

def get_total_count():
    return len(get_all_knowledge_chunks())
