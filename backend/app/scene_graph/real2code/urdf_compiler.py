"""
Real2Code: URDF Compiler
========================

V-NEXT URDF generation from kinematic analysis.

Generates valid URDF/Xacro from:
- SPLART rigid parts
- JointDetector joint specifications
- Optionally: LLM-guided parameter suggestions

No mechanism type guessing. Pure structural URDF.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class LinkSpec:
    """Specification for a URDF link."""
    name: str
    mass: float = 1.0
    com: np.ndarray = None  # [3] center of mass (default origin)
    inertia: np.ndarray = None  # [6] ixx, ixy, ixz, iyy, iyz, izz (default identity)
    visual_geometry: str = "box"  # mesh filename or primitive type
    visual_size: List[float] = None  # [3] for box/sphere
    collision_geometry: str = "box"
    collision_size: List[float] = None
    material_color: List[float] = None  # [4] RGBA
    
    def __post_init__(self):
        if self.com is None:
            self.com = np.zeros(3)
        if self.inertia is None:
            self.inertia = np.array([0.001, 0, 0, 0.001, 0, 0.001])
        if self.visual_size is None:
            self.visual_size = [0.1, 0.1, 0.1]
        if self.collision_size is None:
            self.collision_size = self.visual_size
        if self.material_color is None:
            self.material_color = [0.5, 0.5, 0.5, 1.0]


@dataclass
class JointSpec:
    """Specification for a URDF joint."""
    name: str
    parent_link: str
    child_link: str
    joint_type: str  # revolute, prismatic, continuous, fixed, floating, planar
    axis: np.ndarray = None  # [3] axis direction (default [1, 0, 0])
    origin_xyz: np.ndarray = None  # [3] position
    origin_rpy: np.ndarray = None  # [3] rotation (roll, pitch, yaw)
    limits: Dict = None  # {lower, upper, effort, velocity, spring_stiffness}
    dynamics: Dict = None  # {damping, friction}
    safety_controller: Dict = None  # {soft_lower_limit, soft_upper_limit, k_position, k_velocity}
    
    def __post_init__(self):
        if self.axis is None:
            self.axis = np.array([1.0, 0.0, 0.0])
        if self.origin_xyz is None:
            self.origin_xyz = np.zeros(3)
        if self.origin_rpy is None:
            self.origin_rpy = np.zeros(3)
        if self.limits is None:
            self.limits = {
                "lower": -3.14159,
                "upper": 3.14159,
                "effort": 100.0,
                "velocity": 10.0,
            }
        if self.dynamics is None:
            self.dynamics = {"damping": 0.0, "friction": 0.0}


@dataclass
class RobotSpec:
    """Complete robot specification for URDF generation."""
    name: str
    links: List[LinkSpec]
    joints: List[JointSpec]
    world_link: Optional[str] = None  # If specified, add fixed joint to world


class URDFCompiler:
    """
    Compile URDF from robot specification.
    
    Pure structural translation:
    - LinkSpec → <link> element
    - JointSpec → <joint> element
    - No interpretation, no guessing
    """
    
    def __init__(self):
        self.indent = "  "
    
    def compile(self, robot: RobotSpec, format: str = "urdf") -> str:
        """
        Generate URDF XML from RobotSpec.
        
        Args:
            robot: Robot specification
            format: "urdf" or "xacro"
            
        Returns:
            URDF/Xacro XML string
        """
        if format == "urdf":
            return self._compile_urdf(robot)
        elif format == "xacro":
            return self._compile_xacro(robot)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _compile_urdf(self, robot: RobotSpec) -> str:
        """Generate URDF XML."""
        lines = [
            '<?xml version="1.0"?>',
            f'<robot name="{robot.name}">',
            '',
        ]
        
        # Add world link if specified
        if robot.world_link:
            lines.append(self._indent(1) + f'<link name="world"/>')
            lines.append('')
        
        # Links
        lines.append(self._indent(1) + '<!-- LINKS -->')
        for link in robot.links:
            lines.extend(self._compile_link(link, indent=1))
            lines.append('')
        
        # Joints
        lines.append(self._indent(1) + '<!-- JOINTS -->')
        for joint in robot.joints:
            lines.extend(self._compile_joint(joint, indent=1))
            lines.append('')
        
        lines.append('</robot>')
        
        return '\n'.join(lines)
    
    def _compile_xacro(self, robot: RobotSpec) -> str:
        """Generate Xacro XML with macros."""
        lines = [
            '<?xml version="1.0"?>',
            '<robot xmlns:xacro="http://www.ros.org/xacro"',
            '       name="{robot.name}">',
            '',
            '  <!-- Xacro Macros -->',
            '  <xacro:macro name="default_inertial" params="mass:=1.0 origin_xyz:=(0 0 0) ixx:=0.001 ixy:=0 ixz:=0 iyy:=0.001 iyz:=0 izz:=0.001">',
            '    <inertial>',
            '      <origin xyz="${origin_xyz}" rpy="0 0 0"/>',
            '      <mass value="${mass}"/>',
            '      <inertia ixx="${ixx}" ixy="${ixy}" ixz="${ixz}" iyy="${iyy}" iyz="${iyz}" izz="${izz}"/>',
            '    </inertial>',
            '  </xacro:macro>',
            '',
        ]
        
        # Simplified URDF in xacro
        robot_copy = RobotSpec(
            name=robot.name,
            links=robot.links,
            joints=robot.joints,
            world_link=robot.world_link,
        )
        lines.append(self._compile_urdf(robot_copy))
        lines.append('</robot>')
        
        return '\n'.join(lines)
    
    def _compile_link(self, link: LinkSpec, indent: int = 1) -> List[str]:
        """Compile a single link element."""
        i = self._indent(indent)
        lines = [f'{i}<link name="{link.name}">']
        
        # Inertial
        lines.append(f'{i}  <inertial>')
        lines.append(f'{i}    <origin xyz="{link.com[0]:.6f} {link.com[1]:.6f} {link.com[2]:.6f}" rpy="0 0 0"/>')
        lines.append(f'{i}    <mass value="{link.mass:.6f}"/>')
        lines.append(f'{i}    <inertia ixx="{link.inertia[0]:.6f}" ixy="{link.inertia[1]:.6f}" ixz="{link.inertia[2]:.6f}" iyy="{link.inertia[3]:.6f}" iyz="{link.inertia[4]:.6f}" izz="{link.inertia[5]:.6f}"/>')
        lines.append(f'{i}  </inertial>')
        
        # Visual
        lines.append(f'{i}  <visual>')
        lines.append(f'{i}    <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f'{i}    <geometry>')
        
        if link.visual_geometry == "box":
            sx, sy, sz = link.visual_size
            lines.append(f'{i}      <box size="{sx:.4f} {sy:.4f} {sz:.4f}"/>')
        elif link.visual_geometry == "sphere":
            r = link.visual_size[0] if link.visual_size else 0.05
            lines.append(f'{i}      <sphere radius="{r:.4f}"/>')
        elif link.visual_geometry == "cylinder":
            r, l = link.visual_size[0] if link.visual_size else 0.05, link.visual_size[1] if len(link.visual_size) > 1 else 0.1
            lines.append(f'{i}      <cylinder radius="{r:.4f}" length="{l:.4f}"/>')
        elif link.visual_geometry.endswith('.stl') or link.visual_geometry.endswith('.dae'):
            lines.append(f'{i}      <mesh filename="{link.visual_geometry}"/>')
        else:
            lines.append(f'{i}      <box size="0.1 0.1 0.1"/>')
        
        lines.append(f'{i}    </geometry>')
        
        # Material
        if link.material_color:
            r, g, b, a = link.material_color
            lines.append(f'{i}    <material name="">')
            lines.append(f'{i}      <color rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}"/>')
            lines.append(f'{i}    </material>')
        
        lines.append(f'{i}  </visual>')
        
        # Collision
        lines.append(f'{i}  <collision>')
        lines.append(f'{i}    <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f'{i}    <geometry>')
        
        if link.collision_geometry == "box":
            sx, sy, sz = link.collision_size
            lines.append(f'{i}      <box size="{sx:.4f} {sy:.4f} {sz:.4f}"/>')
        elif link.collision_geometry == "sphere":
            r = link.collision_size[0] if link.collision_size else 0.05
            lines.append(f'{i}      <sphere radius="{r:.4f}"/>')
        elif link.collision_geometry == "cylinder":
            r, l = link.collision_size[0] if link.collision_size else 0.05, link.collision_size[1] if len(link.collision_size) > 1 else 0.1
            lines.append(f'{i}      <cylinder radius="{r:.4f}" length="{l:.4f}"/>')
        else:
            lines.append(f'{i}      <box size="0.1 0.1 0.1"/>')
        
        lines.append(f'{i}    </geometry>')
        lines.append(f'{i}  </collision>')
        
        lines.append(f'{i}</link>')
        
        return lines
    
    def _compile_joint(self, joint: JointSpec, indent: int = 1) -> List[str]:
        """Compile a single joint element."""
        i = self._indent(indent)
        lines = [f'{i}<joint name="{joint.name}" type="{joint.joint_type}">']
        
        # Parent and child
        lines.append(f'{i}  <parent link="{joint.parent_link}"/>')
        lines.append(f'{i}  <child link="{joint.child_link}"/>')
        
        # Origin
        xyz = joint.origin_xyz
        rpy = joint.origin_rpy
        lines.append(f'{i}  <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>')
        
        # Axis (not for fixed joints)
        if joint.joint_type != "fixed":
            ax, ay, az = joint.axis
            lines.append(f'{i}  <axis xyz="{ax:.6f} {ay:.6f} {az:.6f}"/>')
        
        # Limits
        if joint.joint_type in ["revolute", "prismatic"]:
            limits = joint.limits
            lines.append(f'{i}  <limit lower="{limits.get("lower", -3.14):.6f}" upper="{limits.get("upper", 3.14):.6f}" effort="{limits.get("effort", 100):.2f}" velocity="{limits.get("velocity", 10):.2f}"/>')
        
        # Dynamics
        if joint.dynamics.get("damping") or joint.dynamics.get("friction"):
            dyn = joint.dynamics
            lines.append(f'{i}  <dynamics damping="{dyn.get("damping", 0):.6f}" friction="{dyn.get("friction", 0):.6f}"/>')
        
        lines.append(f'{i}</joint>')
        
        return lines
    
    def _indent(self, level: int) -> str:
        return self.indent * level
    
    def from_kinematic_tree(
        self,
        parts: List[Dict],  # [{name, trajectory, gaussian_cloud}]
        joints: List[Dict],  # [{parent, child, type, axis}]
        robot_name: str = "aether_robot",
    ) -> str:
        """
        Compile URDF from kinematic tree data.
        
        Args:
            parts: List of rigid part specs
            joints: List of joint specs
            robot_name: Name for the robot
            
        Returns:
            URDF XML string
        """
        # Convert to RobotSpec
        links = []
        for part in parts:
            link = LinkSpec(
                name=part["name"],
                mass=part.get("mass", 1.0),
                visual_size=part.get("size", [0.1, 0.1, 0.1]),
                material_color=part.get("color", [0.5, 0.5, 0.8, 1.0]),
            )
            links.append(link)
        
        joint_specs = []
        for j in joints:
            joint = JointSpec(
                name=j["name"],
                parent_link=j["parent"],
                child_link=j["child"],
                joint_type=j["type"],
                axis=j.get("axis", np.array([0.0, 0.0, 1.0])),
                origin_xyz=j.get("origin", np.zeros(3)),
                limits=j.get("limits", {"lower": -3.14, "upper": 3.14}),
            )
            joint_specs.append(joint)
        
        robot = RobotSpec(
            name=robot_name,
            links=links,
            joints=joint_specs,
            world_link="world",
        )
        
        return self.compile(robot)


def test_urdf_compiler():
    """Test URDF compilation."""
    print("=" * 60)
    print("Testing URDF Compiler")
    print("=" * 60)
    
    compiler = URDFCompiler()
    
    # Create simple 2-link robot
    robot = RobotSpec(
        name="test_robot",
        links=[
            LinkSpec(
                name="base_link",
                mass=1.0,
                visual_size=[0.2, 0.2, 0.1],
                material_color=[0.8, 0.2, 0.2, 1.0],  # Red
            ),
            LinkSpec(
                name="link1",
                mass=0.5,
                visual_size=[0.1, 0.1, 0.3],
                material_color=[0.2, 0.8, 0.2, 1.0],  # Green
            ),
            LinkSpec(
                name="link2",
                mass=0.3,
                visual_size=[0.08, 0.08, 0.2],
                material_color=[0.2, 0.2, 0.8, 1.0],  # Blue
            ),
        ],
        joints=[
            JointSpec(
                name="joint1",
                parent_link="world",
                child_link="base_link",
                joint_type="fixed",
            ),
            JointSpec(
                name="joint2",
                parent_link="base_link",
                child_link="link1",
                joint_type="revolute",
                axis=np.array([0.0, 0.0, 1.0]),
                limits={"lower": -3.14, "upper": 3.14, "effort": 100, "velocity": 10},
            ),
            JointSpec(
                name="joint3",
                parent_link="link1",
                child_link="link2",
                joint_type="revolute",
                axis=np.array([0.0, 1.0, 0.0]),  # Rotate around y-axis
                limits={"lower": -1.57, "upper": 1.57, "effort": 50, "velocity": 5},
            ),
        ],
        world_link="world",
    )
    
    urdf = compiler.compile(robot)
    
    print("\n✅ Generated URDF:")
    print("-" * 40)
    print(urdf[:1000] + "..." if len(urdf) > 1000 else urdf)
    
    # Validate
    assert "</robot>" in urdf
    assert '<robot name="test_robot">' in urdf
    assert '<joint name="joint2" type="revolute">' in urdf
    assert '<link name="base_link">' in urdf
    
    print("\n✅ URDF validation passed!")
    
    print("\n" + "=" * 60)
    print("URDF Compiler: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_urdf_compiler()
