"""ROCG-PA SceneGraph schema - core data model for AETHER."""
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class JointType(str, Enum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    SPRING = "spring"
    BELT = "belt"
    CONTACT = "contact"


class ObjectNode(BaseModel):
    id: str
    label: str
    # Universal object types for ANY mechanism (drone, vehicle, human, robot, etc.)
    object_type: Literal[
        "rigid", "joint", "belt", "pulley", "rail", "frame",
        "carriage", "motor", "propeller", "rotor",
        "torso", "limb", "foot", "head", "hand",
        "chassis", "wheel", "suspension", "axle",
        "link", "end_effector", "base",
        "bob", "rod", "string",
        "particle", "ground", "contact_patch",
        "custom",
    ] = "rigid"
    keypoints: dict = Field(default_factory=lambda: {"canonical": [], "current": []})
    velocity_linear: list = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity_angular: list = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    physics: dict = Field(default_factory=lambda: {"mass_kg": 1.0, "friction": 0.3, "damping": 0.05})
    material_embedding: list = Field(default_factory=lambda: [0.0] * 128)
    latent_state: list = Field(default_factory=lambda: [0.0] * 128)
    uncertainty: dict = Field(default_factory=lambda: {"position_sigma": 0.01, "velocity_sigma": 0.05})
    editable_params: dict = Field(default_factory=lambda: {
        "mass_kg": {"min": 0.01, "max": 100.0, "value": 1.0},
        "friction": {"min": 0.0, "max": 1.0, "value": 0.3},
    })


class Edge(BaseModel):
    source_id: str
    target_id: str
    joint_type: JointType = JointType.CONTACT
    joint_axis: list = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    joint_limits: dict = Field(default_factory=lambda: {"min": None, "max": None})
    contact_prob: float = 0.0
    normal_force: list = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    belt_tension: float = 10.0
    pulley_radius: float = 0.02
    editable: bool = True
    uncertainty: float = 0.1


class CameraIntrinsics(BaseModel):
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0
    distortion_coeffs: list = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    resolution: list = Field(default_factory=lambda: [1920, 1080])
    calibration_date: str = Field(default_factory=lambda: datetime.now().isoformat())
    method: Literal["checkerboard", "automatic", "estimated"] = "estimated"


class ROCGPA_SceneGraph(BaseModel):
    scene_id: str
    session_id: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    camera_intrinsics: CameraIntrinsics | None = None
    objects: list[ObjectNode] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    frame_source: str | None = None
    reconstruction_confidence: float = 0.5
    processing_info: dict = Field(default_factory=lambda: {
        "segmentation_model": "sam2",
        "tracking_model": "cotracker",
    })
    schema_version: str = "1.0"

    def get_object(self, object_id: str) -> ObjectNode | None:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def summary(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "object_count": len(self.objects),
            "edge_count": len(self.edges),
            "object_types": list(set(o.object_type for o in self.objects)),
            "confidence": self.reconstruction_confidence,
        }
