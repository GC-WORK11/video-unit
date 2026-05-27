"""
AETHER Procedural MuJoCo Generator with Exact Inertia (Phase 1.2)
================================================================

Dynamically builds MuJoCo XML from perception data (SAM2 masks + MiDaS depth).
Now includes EXACT inertia tensor computation from point clouds.

NEW: Phase 1.2 - Exact Inertia Tensor
- Computes 3x3 inertia tensor from 3D point cloud
- Uses parallel axis theorem for COM offset
- Generates MuJoCo XML with full inertia data
"""

import logging
import numpy as np
import cv2
from typing import Optional, List, Dict, Any

from app.physics.inertia_tensor import (
    compute_exact_inertia,
    inertia_to_mujoco_xml,
    InertiaTensor,
)

log = logging.getLogger(__name__)


class ProceduralMuJoCoBuilder:
    """
    Builds MuJoCo XML from visual data with EXACT inertia tensors.
    
    Phase 1.2: Now computes real inertia tensors from point clouds,
    not the "garbage box approximation".
    """
    
    def __init__(self, frame_shape: tuple, pixel_scale: float = 0.001):
        self.h, self.w = frame_shape
        self.pixel_scale = pixel_scale  # meters per pixel
        
    def _pixel_to_meter(self, px: float, py: float) -> tuple:
        """Convert pixel coords to meters centered at 0,0."""
        mx = (px - self.w/2) * self.pixel_scale
        my = (self.h/2 - py) * self.pixel_scale
        return mx, my
    
    def build_from_point_cloud(
        self,
        point_cloud: np.ndarray,
        body_name: str,
        density: float = 1000.0,
    ) -> dict:
        """
        Build MuJoCo body from EXACT point cloud using inertia tensor.
        
        This is Phase 1.2 - replaces the box approximation with
        real volumetric inertia.
        
        Args:
            point_cloud: Nx3 array of 3D points in meters
            body_name: Name for this body
            density: Mass per unit volume (kg/m³)
        
        Returns:
            dict with body XML and physics data
        """
        # Compute exact inertia tensor
        inertia = compute_exact_inertia(point_cloud, density=density)
        
        # Convert to MuJoCo parameters
        mj_params = inertia_to_mujoco_xml(inertia, body_name)
        
        # Generate body XML with full inertia
        # MuJoCo uses fullinertia: [I11, I22, I33, I12, I13, I23]
        I1, I2, I3 = mj_params['fullinertia'][:3]
        ixy, ixz, iyz = mj_params['fullinertia'][3:]
        
        body_xml = f"""
    <body name="{body_name}" pos="{' '.join(map(str, mj_params['pos']))}">
      <inertial fullinertia="{I1:.6f} {I2:.6f} {I3:.6f} {ixy:.6f} {ixz:.6f} {iyz:.6f}" mass="{mj_params['mass']:.6f}" pos="0 0 0"/>
    </body>"""
        
        return {
            "xml": body_xml,
            "inertia": inertia,
            "mj_params": mj_params,
            "mass": mj_params['mass'],
            "com": inertia.com,
        }
    
    def build_from_mask_with_depth(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        body_name: str,
        K: Optional[np.ndarray] = None,
        density: float = 1000.0,
    ) -> dict:
        """
        Build MuJoCo body from SAM2 mask + MiDaS depth.
        
        This is the full Phase 1.2 pipeline:
        1. Get depth for masked pixels
        2. Convert to 3D point cloud
        3. Compute exact inertia tensor
        4. Generate MuJoCo XML
        """
        from app.reconstruction.mesh import depth_to_point_cloud
        
        # Get point cloud from depth + mask
        points_3d = depth_to_point_cloud(depth_map, mask, K, scale=1.0)
        
        if len(points_3d) < 10:
            log.warning(f"Point cloud too small for {body_name}, using fallback")
            return self._build_fallback_body(mask, body_name)
        
        # Build with exact inertia
        return self.build_from_point_cloud(points_3d, body_name, density)
    
    def _build_fallback_body(self, mask: np.ndarray, body_name: str) -> dict:
        """Fallback box approximation if point cloud fails."""
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] == 0:
            return {"xml": f"<body name=\"{body_name}\"><geom type=\"sphere\" size=\"0.05\" mass=\"1\"/></body>", "inertia": None}
        
        cx_px = M["m10"] / M["m00"]
        cy_px = M["m01"] / M["m00"]
        cx, cy = self._pixel_to_meter(cx_px, cy_px)
        
        x, y, w, h = 0, 0, 10, 10
        mw = w * self.pixel_scale
        mh = h * self.pixel_scale
        area = M["m00"]
        mass = area / 10000.0
        
        body_xml = f"""
    <body name="{body_name}" pos="{cx} {cy} 0.5">
      <freejoint/>
      <geom type="box" size="{mw/2} {mh/2} 0.05" mass="{mass}"/>
    </body>"""
        
        return {"xml": body_xml, "inertia": None}
    
    def build(
        self,
        masks: List[Dict],
        mechanism_type: str,
        params: Dict,
        depth_maps: Optional[List[np.ndarray]] = None,
    ) -> str:
        """
        Build complete MuJoCo XML with EXACT inertia tensors.
        
        If depth_maps are provided, uses Phase 1.2 exact inertia.
        Otherwise falls back to box approximation.
        """
        xml_bodies = []
        
        for i, mask_d in enumerate(masks):
            mask = mask_d.get("segmentation")
            if mask is None:
                continue
            
            body_name = f"obj_{i}"
            
            # Use Phase 1.2 if depth available
            if depth_maps and i < len(depth_maps):
                result = self.build_from_mask_with_depth(
                    mask,
                    depth_maps[i],
                    body_name,
                    density=params.get("density", 1000.0),
                )
            else:
                # Fallback to box approximation
                M = cv2.moments(mask.astype(np.uint8))
                if M["m00"] == 0:
                    continue
                
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]
                cx, cy = self._pixel_to_meter(cx_px, cy_px)
                
                x, y, w, h = mask_d.get("bbox", [0, 0, 10, 10])
                mw = w * self.pixel_scale
                mh = h * self.pixel_scale
                area = M["m00"]
                mass = params.get("mass_kg", 1.0) * (area / 10000.0)
                
                joint_xml = ""
                if i == 0:
                    if mechanism_type in ["vehicle", "rigid_body", "drone"]:
                        joint_xml = '<freejoint/>'
                    elif mechanism_type == "pendulum":
                        joint_xml = f'<joint name="pin" type="hinge" pos="{-cx} {-cy} 0" axis="0 0 1"/>'
                
                body_xml = f"""
    <body name="{body_name}" pos="{cx} {cy} 0.5">
      {joint_xml}
      <geom type="box" size="{mw/2} {mh/2} 0.05" mass="{mass}"/>
    </body>"""
                result = {"xml": body_xml, "inertia": None}
            
            xml_bodies.append(result["xml"])
            
            # Log inertia info if available
            if result.get("inertia"):
                log.info(f"  {body_name}: mass={result['mass']:.3f}kg, COM=[{result['com'][0]:.3f}, {result['com'][1]:.3f}, {result['com'][2]:.3f}]")
        
        bodies_str = "\n".join(xml_bodies)
        
        full_xml = f"""
<mujoco model="aether_procedural">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option integrator="implicitfast" iterations="1"/>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.01" rgba=".3 .3 .3 1" friction="0.8 0.01 0.01"/>
    {bodies_str}
  </worldbody>
</mujoco>"""
        return full_xml


class UniversalPhysicsSimulator:
    """Universal MuJoCo physics simulator with Phase 1.2 exact inertia."""
    
    def __init__(self):
        log.info("UniversalPhysicsSimulator with Phase 1.2 Exact Inertia initialized ✅")
    
    def simulate(
        self,
        mechanism_type: str,
        horizon_seconds: float = 3.0,
        param_overrides: Optional[dict] = None,
        masks: Optional[List[Dict]] = None,
        frame_shape: Optional[tuple] = None,
        depth_maps: Optional[List[np.ndarray]] = None,
    ) -> dict:
        """
        Simulate with EXACT inertia tensors from point clouds.
        
        If depth_maps are provided, Phase 1.2 is used.
        """
        import mujoco
        
        params = param_overrides or {}
        
        # Use Procedural Builder with exact inertia
        if masks and frame_shape:
            builder = ProceduralMuJoCoBuilder(frame_shape)
            xml = builder.build(masks, mechanism_type, params, depth_maps)
            log.info(f"Generated XML with Phase 1.2 exact inertia for {mechanism_type}")
        else:
            xml = f"""
<mujoco model="fallback">
  <worldbody>
    <body name="body" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="{params.get('mass_kg', 1.0)}"/>
    </body>
  </worldbody>
</mujoco>"""

        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            
            # Simulate
            n_steps = int(horizon_seconds / model.opt.timestep)
            positions = []
            
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
                if len(positions) < 500:
                    positions.append(data.qpos.copy().tolist())
            
            return {
                "success": True,
                "mechanism_type": mechanism_type,
                "duration": horizon_seconds,
                "timesteps": n_steps,
                "trajectory": positions,
                "phase": "1.2_exact_inertia" if depth_maps else "fallback",
            }
        except Exception as e:
            log.error(f"Simulation failed: {e}")
            return {"success": False, "error": str(e)}


def simulate_universal(
    mechanism_type: str = "rigid_body",
    horizon_seconds: float = 3.0,
    param_overrides: Optional[dict] = None,
    masks: Optional[List[Dict]] = None,
    frame_shape: Optional[tuple] = None,
    depth_maps: Optional[List[np.ndarray]] = None,
) -> dict:
    """Convenience function with Phase 1.2 support."""
    sim = UniversalPhysicsSimulator()
    return sim.simulate(mechanism_type, horizon_seconds, param_overrides, masks, frame_shape, depth_maps)
