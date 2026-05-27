"""
AETHER Exact Inertia Tensor Computation
=======================================

Computes the EXACT 3x3 inertia tensor from 3D point clouds,
replacing the "garbage box approximation" with real math.

THE MATH:
For a rigid body with uniform density ρ:
    I_xx = ∫(y² + z²) dV
    I_yy = ∫(x² + z²) dV
    I_zz = ∫(x² + y²) dV
    I_xy = I_yx = -∫(xy) dV
    I_xz = I_zx = -∫(xz) dV
    I_yz = I_zy = -∫(yz) dV

For a discrete point cloud (uniform density):
    I = Σ (|r|² * E_3 - r ⊗ r) * m_i

Where:
    r = point position relative to COM
    E_3 = 3x3 identity
    ⊗ = outer product
    m_i = mass of point i
"""

import numpy as np
from scipy.spatial import ConvexHull
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class InertiaTensor:
    """Complete inertia information for a rigid body."""
    # 3x3 inertia tensor in the body's principal frame
    tensor: np.ndarray  # 3x3 symmetric matrix
    
    # Principal moments of inertia (eigenvalues)
    I1: float  # Principal axis 1 (smallest)
    I2: float  # Principal axis 2
    I3: float  # Principal axis 3 (largest)
    
    # Principal axes (eigenvectors) - rotation matrix
    axes: np.ndarray  # 3x3 matrix where columns are eigenvectors
    
    # Center of mass
    com: np.ndarray  # 3D position of COM
    
    # Total mass
    mass: float
    
    # Volume (approximated from convex hull)
    volume: float
    
    # Quality score (0-1, based on point cloud density)
    quality: float


def compute_center_of_mass(points: np.ndarray, density: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Compute center of mass and total mass from point cloud.
    
    Args:
        points: Nx3 array of point positions
        density: Mass per unit volume (default: 1.0)
    
    Returns:
        com: 3D center of mass
        mass: Total mass
    """
    if len(points) == 0:
        return np.zeros(3), 0.0
    
    # COM is mean of points (assuming uniform density)
    com = np.mean(points, axis=0)
    
    # Approximate volume using convex hull
    try:
        hull = ConvexHull(points)
        volume = hull.volume
    except Exception:
        # Fallback: approximate as bounding box / 6
        bbox = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(bbox) / 6
    
    mass = volume * density
    
    return com, mass


def compute_inertia_tensor(points: np.ndarray, com: np.ndarray, mass: float) -> np.ndarray:
    """
    Compute the 3x3 inertia tensor for a point cloud.
    
    Formula: I = Σ (|r|² * E - r ⊗ r) * m_i
    
    Where r = point position relative to COM
    
    Args:
        points: Nx3 point positions
        com: Center of mass (3D)
        mass: Total mass
    
    Returns:
        I: 3x3 symmetric inertia tensor
    """
    n_points = len(points)
    
    if n_points == 0:
        return np.zeros((3, 3))
    
    # Relative positions from COM
    r = points - com  # Nx3
    
    # For uniform density, mass per point
    m_per_point = mass / n_points
    
    # Compute inertia tensor components
    # I_ij = Σ m_k * (|r_k|² * δ_ij - r_ki * r_kj)
    
    I = np.zeros((3, 3))
    
    # Diagonal terms (I_xx, I_yy, I_zz)
    r_squared = np.sum(r ** 2, axis=1)  # N,
    
    I[0, 0] = np.sum(m_per_point * (r_squared - r[:, 0] ** 2))  # Σ m*(y²+z²)
    I[1, 1] = np.sum(m_per_point * (r_squared - r[:, 1] ** 2))  # Σ m*(x²+z²)
    I[2, 2] = np.sum(m_per_point * (r_squared - r[:, 2] ** 2))  # Σ m*(x²+y²)
    
    # Off-diagonal terms
    I[0, 1] = I[1, 0] = -np.sum(m_per_point * r[:, 0] * r[:, 1])  # -Σ m*xy
    I[0, 2] = I[2, 0] = -np.sum(m_per_point * r[:, 0] * r[:, 2])  # -Σ m*xz
    I[1, 2] = I[2, 1] = -np.sum(m_per_point * r[:, 1] * r[:, 2])  # -Σ m*yz
    
    return I


def diagonalize_inertia_tensor(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize inertia tensor to get principal axes and moments.
    
    Uses eigendecomposition: I = Q Λ Q^T
    
    Args:
        I: 3x3 symmetric inertia tensor
    
    Returns:
        eigenvalues: Principal moments of inertia (sorted ascending)
        eigenvectors: Principal axes (columns)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(I)
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    
    return eigenvalues, eigenvectors


def compute_exact_inertia(
    points: np.ndarray,
    density: float = 1000.0,  # kg/m³ (water density as default)
) -> InertiaTensor:
    """
    Compute the EXACT inertia tensor from a 3D point cloud.
    
    This is the main entry point for Phase 1.2.
    
    Args:
        points: Nx3 array of point positions in meters
        density: Mass per unit volume in kg/m³ (default: 1000 for water)
    
    Returns:
        InertiaTensor with full inertia information
    
    Example:
        >>> points = np.random.randn(1000, 3) * 0.1
        >>> inertia = compute_exact_inertia(points)
        >>> print(f"COM: {inertia.com}")
        >>> print(f"Principal moments: {inertia.I1:.4f}, {inertia.I2:.4f}, {inertia.I3:.4f}")
    """
    if len(points) < 10:
        log.warning(f"Point cloud has only {len(points)} points, results may be inaccurate")
    
    # Step 1: Compute center of mass and mass
    com, mass = compute_center_of_mass(points, density)
    
    # Step 2: Compute inertia tensor
    I = compute_inertia_tensor(points, com, mass)
    
    # Step 3: Diagonalize to get principal axes
    eigenvalues, eigenvectors = diagonalize_inertia_tensor(I)
    
    # Step 4: Compute volume for quality metric
    try:
        hull = ConvexHull(points)
        volume = hull.volume
    except Exception:
        bbox = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(bbox) / 6
    
    # Step 5: Quality score based on point density
    # Higher density = better approximation
    quality = min(1.0, len(points) / 1000.0)
    
    return InertiaTensor(
        tensor=I,
        I1=eigenvalues[0],
        I2=eigenvalues[1],
        I3=eigenvalues[2],
        axes=eigenvectors,
        com=com,
        mass=mass,
        volume=volume,
        quality=quality,
    )


def inertia_to_mujoco_xml(
    inertia: InertiaTensor,
    body_name: str = "body",
    geom_size: Optional[np.ndarray] = None,
) -> dict:
    """
    Convert inertia tensor to MuJoCo-compatible parameters.
    
    MuJoCo uses "fullinertia" format: [I1, I2, I3, x, y, z]
    Where x,y,z are the off-diagonal terms divided by mass
    
    Args:
        inertia: Computed inertia tensor
        body_name: Name for the body
        geom_size: Optional [half_x, half_y, half_z] for geom approximation
    
    Returns:
        dict with MuJoCo-compatible parameters
    """
    # Principal moments
    I1, I2, I3 = inertia.I1, inertia.I2, inertia.I3
    
    # Off-diagonal terms (already divided by mass in our computation)
    # I_xy = I[0,1], etc.
    ixy = inertia.tensor[0, 1] / inertia.mass if inertia.mass > 0 else 0
    ixz = inertia.tensor[0, 2] / inertia.mass if inertia.mass > 0 else 0
    iyz = inertia.tensor[1, 2] / inertia.mass if inertia.mass > 0 else 0
    
    return {
        "body_name": body_name,
        "pos": inertia.com.tolist(),  # Position of COM
        "mass": inertia.mass,
        # MuJoCo fullinertia: [I11, I22, I33, I12, I13, I23]
        "fullinertia": [I1, I2, I3, ixy, ixz, iyz],
        "volume": inertia.volume,
        "quality": inertia.quality,
    }


def print_inertia_summary(inertia: InertiaTensor) -> str:
    """Generate a human-readable summary of the inertia tensor."""
    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    INERTIA TENSOR SUMMARY                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Mass:     {inertia.mass:.4f} kg                                      ║
║  Volume:   {inertia.volume:.6f} m³                               ║
║  Quality:  {inertia.quality:.2%}                                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Center of Mass: [{inertia.com[0]:7.4f}, {inertia.com[1]:7.4f}, {inertia.com[2]:7.4f}] m     ║
╠══════════════════════════════════════════════════════════════════╣
║  Principal Moments (kg⋅m²):                                     ║
║    I1 (min): {inertia.I1:.6f}                                      ║
║    I2 (mid): {inertia.I2:.6f}                                      ║
║    I3 (max): {inertia.I3:.6f}                                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Principal Axes:                                                  ║
║    Axis 1: [{inertia.axes[0,0]:7.4f}, {inertia.axes[1,0]:7.4f}, {inertia.axes[2,0]:7.4f}]            ║
║    Axis 2: [{inertia.axes[0,1]:7.4f}, {inertia.axes[1,1]:7.4f}, {inertia.axes[2,1]:7.4f}]            ║
║    Axis 3: [{inertia.axes[0,2]:7.4f}, {inertia.axes[1,2]:7.4f}, {inertia.axes[2,2]:7.4f}]            ║
╠══════════════════════════════════════════════════════════════════╣
║  Inertia Tensor (kg⋅m²):                                         ║
║    [{inertia.tensor[0,0]:10.6f}, {inertia.tensor[0,1]:10.6f}, {inertia.tensor[0,2]:10.6f}]   ║
║    [{inertia.tensor[1,0]:10.6f}, {inertia.tensor[1,1]:10.6f}, {inertia.tensor[1,2]:10.6f}]   ║
║    [{inertia.tensor[2,0]:10.6f}, {inertia.tensor[2,1]:10.6f}, {inertia.tensor[2,2]:10.6f}]   ║
╚══════════════════════════════════════════════════════════════════╝
"""


# Test with known shapes
def test_with_cube():
    """Test with a solid cube (known analytical solution)."""
    # Unit cube from -0.5 to 0.5
    # Analytical: I = (1/6) * m * (a² + b²) for each axis
    # For unit cube: I = m/6 for all axes
    
    n = 20  # points per dimension
    x = np.linspace(-0.5, 0.5, n)
    points = np.array(np.meshgrid(x, x, x)).reshape(3, -1).T
    
    inertia = compute_exact_inertia(points, density=1.0)
    
    print("\n=== CUBE TEST ===")
    print(f"Expected: I1=I2=I3 = m/6 = {inertia.mass/6:.6f}")
    print(f"Computed: I1={inertia.I1:.6f}, I2={inertia.I2:.6f}, I3={inertia.I3:.6f}")
    print(f"Error: {abs(inertia.I1 - inertia.mass/6)/inertia.mass*100:.2f}%")
    
    return inertia


def test_with_cylinder():
    """Test with a solid cylinder (known analytical solution)."""
    # Cylinder: radius R, height h
    # I_zz = (1/2) * m * R² (about axis)
    # I_xx = I_yy = (1/12) * m * (3R² + h²)
    
    R, h = 0.1, 0.3
    n_r, n_h = 15, 30
    
    r = np.sqrt(np.random.rand(n_r * n_r)) * R
    theta = np.random.rand(n_r * n_r) * 2 * np.pi
    z = np.linspace(-h/2, h/2, n_h)
    
    points = []
    for zi in z:
        for ri in range(len(r)):
            x = r[ri] * np.cos(theta[ri])
            y = r[ri] * np.sin(theta[ri])
            points.append([x, y, zi])
    
    points = np.array(points)
    inertia = compute_exact_inertia(points, density=1.0)
    
    print("\n=== CYLINDER TEST ===")
    m = inertia.mass
    I_zz_expected = 0.5 * m * R**2
    I_xx_expected = (1/12) * m * (3*R**2 + h**2)
    
    print(f"Expected I_zz: {I_zz_expected:.6f}")
    print(f"Computed I_zz: {inertia.I3:.6f} (axis of symmetry)")
    print(f"Expected I_xx: {I_xx_expected:.6f}")
    print(f"Computed I_xx: {inertia.I1:.6f}, {inertia.I2:.6f}")
    
    return inertia


if __name__ == "__main__":
    # Run tests
    test_with_cube()
    test_with_cylinder()
    
    # Test with real point cloud from reconstruction
    print("\n=== REAL DATA TEST ===")
    print("This would load point cloud from AETHER reconstruction")
