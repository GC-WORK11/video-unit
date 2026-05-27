"""Scene graph builder — universal, perception-driven scene graph for ANY mechanism.

AETHER is domain-agnostic. It can handle:
  - Belt/gantry systems
  - Drone dynamics
  - Human motion (skateboard, sports)
  - Vehicle dynamics (RC cars, bikes)
  - Any mechanism the user uploads

The scene graph is built from actual perception results (SAM 2 + CoTracker3),
not hardcoded assumptions. Physics parameters are derived from detected objects
and their relationships.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Literal

import numpy as np

from app.core.config import DATA_DIR
from app.scene_graph.schema import (
    ROCGPA_SceneGraph, ObjectNode, Edge, JointType, CameraIntrinsics
)

log = logging.getLogger(__name__)

# Supported mechanism types — AETHER auto-detects or user selects
MECHANISM_TYPES = Literal[
    "belt_gantry", "drone", "human_motion", "vehicle", "robot_arm",
    " linkage", "pendulum", "rigid_body", "custom"
]

# Physics defaults per mechanism type
PHYSICS_DEFAULTS: dict[str, dict] = {
    "belt_gantry": {
        "belt":   {"mass_kg": 0.5,  "friction": 0.3,  "damping": 0.05, "tension": 20.0},
        "carriage": {"mass_kg": 1.2, "friction": 0.15, "damping": 0.08},
        "pulley":  {"mass_kg": 0.3,  "friction": 0.1,  "radius": 0.01},
        "frame":   {"mass_kg": 10.0, "friction": 0.5},
    },
    "drone": {
        "body":        {"mass_kg": 1.0,  "friction": 0.01, "damping": 0.1},
        "motor":       {"mass_kg": 0.1,  "friction": 0.05, "thrust_N_kgf": 0.5},
        "propeller":   {"mass_kg": 0.05, "friction": 0.01, "damping": 0.02},
    },
    "human_motion": {
        "torso":    {"mass_kg": 25.0, "friction": 0.4, "damping": 0.5},
        "limb":     {"mass_kg": 5.0,  "friction": 0.3, "damping": 0.2},
        "foot":     {"mass_kg": 1.5,  "friction": 0.7, "damping": 0.1},
    },
    "vehicle": {
        "chassis": {"mass_kg": 2.0,  "friction": 0.3, "damping": 0.1},
        "wheel":   {"mass_kg": 0.3,  "friction": 0.8, "damping": 0.05, "radius": 0.03},
        "susp":    {"mass_kg": 0.2,  "friction": 0.2, "damping": 0.3, "stiffness": 100.0},
        "motor":   {"mass_kg": 0.5,  "friction": 0.1, "torque_Nm": 1.5},
    },
    "robot_arm": {
        "link":    {"mass_kg": 1.0,  "friction": 0.1,  "damping": 0.05},
        "joint":    {"mass_kg": 0.2,  "friction": 0.2,  "damping": 0.1},
        "end_eff": {"mass_kg": 0.3,  "friction": 0.1,  "damping": 0.02},
    },
    "linkage": {
        "link":  {"mass_kg": 0.5, "friction": 0.1, "damping": 0.05},
        "pivot": {"mass_kg": 0.1, "friction": 0.2, "damping": 0.05},
    },
    "pendulum": {
        "bob":  {"mass_kg": 1.0, "friction": 0.01, "damping": 0.02},
        "rod":  {"mass_kg": 0.1, "friction": 0.01, "damping": 0.01},
    },
    "rigid_body": {
        "body": {"mass_kg": 1.0, "friction": 0.3, "damping": 0.1},
    },
    "custom": {
        "object": {"mass_kg": 1.0, "friction": 0.3, "damping": 0.1},
    },
}

# Joint types appropriate per mechanism
MECHANISM_JOINTS: dict[str, list[JointType]] = {
    "belt_gantry":  [JointType.BELT, JointType.PRISMATIC, JointType.FIXED],
    "drone":        [JointType.FIXED, JointType.REVOLUTE],
    "human_motion": [JointType.REVOLUTE, JointType.CONTACT],
    "vehicle":      [JointType.PRISMATIC, JointType.REVOLUTE, JointType.FIXED, JointType.CONTACT],
    "robot_arm":    [JointType.REVOLUTE, JointType.PRISMATIC, JointType.FIXED],
    "linkage":      [JointType.REVOLUTE, JointType.PRISMATIC, JointType.FIXED],
    "pendulum":     [JointType.REVOLUTE, JointType.FIXED],
    "rigid_body":   [JointType.CONTACT, JointType.FIXED],
    "custom":       [JointType.CONTACT, JointType.FIXED, JointType.REVOLUTE, JointType.PRISMATIC],
}


def build_scene_graph(
    session_id: str,
    mechanism_type: str = "auto",
    object_labels: list[str] | None = None,
    keypoint_data: list[dict] | None = None,
    detected_objects: list[dict] | None = None,
) -> ROCGPA_SceneGraph:
    """Build a SceneGraph for ANY mechanism type.

    This is the universal entry point. It works in two modes:

    1. **Perception-driven** (detected_objects provided):
       Uses actual SAM 2 + CoTracker3 results to build the scene graph.
       Auto-detects object types from their appearance/motion.

    2. **Template-based** (detected_objects not provided):
       Creates a physics-grounded template for the specified mechanism_type.
       User can then edit object labels and parameters.

    Args:
        session_id:     The session ID for data paths
        mechanism_type: "auto" | "belt_gantry" | "drone" | "human_motion" |
                       "vehicle" | "robot_arm" | "linkage" | "pendulum" |
                       "rigid_body" | "custom"
        object_labels:  Human-readable names for detected objects
        keypoint_data:  CoTracker3 keypoint trajectories per object
        detected_objects: Full perception results from SAM 2
    """
    scene_id = str(uuid.uuid4())[:8]

    # Check for perception results
    perception_path = DATA_DIR / "sessions" / session_id / "perception" / "perception.json"
    has_perception = perception_path.exists()

    # Load perception data if available
    perception_data = None
    if has_perception:
        try:
            with open(perception_path) as f:
                perception_data = json.load(f)
            log.info(f"Loaded perception data for session {session_id}: "
                     f"{len(perception_data.get('objects', []))} objects detected")
        except Exception as e:
            log.warning(f"Failed to load perception data: {e}")
            has_perception = False

    # Camera intrinsics
    camera = CameraIntrinsics(
        fx=950.0, fy=950.0, cx=640.0, cy=360.0,
        resolution=[1280, 720], method="estimated",
    )

    # Determine mechanism type
    detected_mechanism = None
    if mechanism_type == "auto":
        if perception_data:
            detected_mechanism = _infer_mechanism_type(perception_data)
        else:
            detected_mechanism = "rigid_body"
        log.info(f"Auto-detected mechanism type: {detected_mechanism}")
    else:
        detected_mechanism = mechanism_type

    # Build the graph
    if detected_objects and perception_data:
        objects, edges = _build_from_perception(
            perception_data, detected_mechanism, object_labels or []
        )
    elif has_perception and perception_data:
        objects, edges = _build_from_perception(
            perception_data, detected_mechanism, object_labels or []
        )
    else:
        objects, edges = _build_from_template(detected_mechanism, object_labels or [])

    scene = ROCGPA_SceneGraph(
        scene_id=scene_id,
        session_id=session_id,
        camera_intrinsics=camera,
        objects=objects,
        edges=edges,
        frame_source=f"sessions/{session_id}/frames",
        reconstruction_confidence=0.8 if has_perception else 0.5,
        processing_info={
            "segmentation_model": "sam2" if has_perception else "none",
            "tracking_model": "cotracker3" if has_perception else "none",
            "depth_model": "depth_anything_v2" if has_perception else "none",
            "perception_available": has_perception,
            "mechanism_type": detected_mechanism,
            "object_count": len(objects),
        },
    )

    return scene


def _infer_mechanism_type(perception_data: dict) -> str:
    """Infer mechanism type from perception data using heuristics."""
    objects = perception_data.get("objects", [])
    if not objects:
        return "rigid_body"

    # Count object types by label
    labels = [o.get("label", "").lower() for o in objects]

    # Heuristic detection
    if any(l in labels for l in ["motor", "propeller", "drone", "quadcopter", "rotor"]):
        return "drone"
    if any(l in labels for l in ["chassis", "wheel", "suspension", "car", "rc car"]):
        return "vehicle"
    if any(l in labels for l in ["torso", "foot", "limb", "human", "skateboard"]):
        return "human_motion"
    if any(l in labels for l in ["link", "arm", "joint", "robot"]):
        return "robot_arm"
    if any(l in labels for l in ["belt", "carriage", "pulley", "gantry", "extruder"]):
        return "belt_gantry"
    if any(l in labels for l in ["bob", "pendulum", "string"]):
        return "pendulum"
    if len(objects) >= 3 and any("link" in l for l in labels):
        return "linkage"

    return "rigid_body"


def _build_from_perception(
    perception_data: dict,
    mechanism_type: str,
    user_labels: list[str],
) -> tuple[list[ObjectNode], list[Edge]]:
    """Build scene graph from actual SAM 2 + CoTracker3 perception results."""
    objects_raw = perception_data.get("objects", [])
    tracks = perception_data.get("tracks", [])

    if not objects_raw:
        return _build_from_template(mechanism_type, user_labels)

    physics_defaults = PHYSICS_DEFAULTS.get(mechanism_type, PHYSICS_DEFAULTS["rigid_body"])
    object_nodes: list[ObjectNode] = []
    object_map: dict[str, ObjectNode] = {}

    for i, obj in enumerate(objects_raw):
        obj_id = obj.get("id", f"obj_{i}")
        label = obj.get("label", user_labels[i] if i < len(user_labels) else f"Object {i+1}")

        # Determine physics params from object type
        obj_type_key = _match_object_type(label, mechanism_type)
        phys = physics_defaults.get(obj_type_key, physics_defaults.get("object", physics_defaults["rigid_body"]))

        # Extract keypoints from CoTracker3 if available
        kp_current = []
        for track in tracks:
            if track.get("object_id") == obj_id or track.get("id") == obj_id:
                trajectory = track.get("trajectory", [])
                if trajectory:
                    kp_current = trajectory[-1] if len(trajectory) > 0 else trajectory[0]
                    break

        # Build editable params based on physics
        editable = {}
        if "mass_kg" in phys:
            editable["mass_kg"] = {"min": 0.1, "max": 100.0, "value": phys["mass_kg"]}
        if "friction" in phys:
            editable["friction"] = {"min": 0.0, "max": 1.0, "value": phys["friction"]}
        if "damping" in phys:
            editable["damping"] = {"min": 0.0, "max": 1.0, "value": phys["damping"]}
        if "tension" in phys:
            editable["belt_tension"] = {"min": 0.0, "max": 200.0, "value": phys["tension"]}
        if "stiffness" in phys:
            editable["stiffness"] = {"min": 1.0, "max": 1000.0, "value": phys["stiffness"]}
        if "thrust_N_kgf" in phys:
            editable["thrust_N"] = {"min": 0.0, "max": 20.0, "value": phys["thrust_N_kgf"] * 9.81}
        if "torque_Nm" in phys:
            editable["torque_Nm"] = {"min": 0.0, "max": 10.0, "value": phys["torque_Nm"]}

        node = ObjectNode(
            id=obj_id,
            label=label,
            object_type=obj.get("type", obj_type_key),
            keypoints={
                "canonical": obj.get("keypoints", [[0, 0, 0]]),
                "current": kp_current if kp_current else [[0, 0, 0]],
            },
            physics=phys,
            editable_params=editable,
        )
        object_nodes.append(node)
        object_map[obj_id] = node

    # Build edges from perception relationships
    edges = _build_edges_from_relationships(perception_data, object_map, mechanism_type)

    return object_nodes, edges


def _match_object_type(label: str, mechanism: str) -> str:
    """Match a label to a known physics object type."""
    label_lower = label.lower()
    mapping = {
        "belt": "belt", "timing_belt": "belt",
        "carriage": "carriage", "slider": "carriage",
        "pulley": "pulley", "idler": "pulley",
        "frame": "frame", "base": "frame", "structure": "frame",
        "motor": "motor", "servo": "motor",
        "propeller": "propeller", "rotor": "propeller",
        "chassis": "chassis", "body": "chassis", "frame_veh": "chassis",
        "wheel": "wheel", "tire": "wheel",
        "suspension": "suspension", "shock": "suspension",
        "torso": "torso", "body_human": "torso",
        "foot": "foot", "shoe": "foot",
        "limb": "limb", "arm": "limb", "leg": "limb",
        "link": "link", "arm_robot": "link",
        "joint": "joint", "pivot": "joint",
        "end_effector": "end_effector", "gripper": "end_effector",
        "bob": "bob", "mass": "bob",
        "rod": "rod", "string": "rod",
    }
    for key, value in mapping.items():
        if key in label_lower:
            return value
    return "rigid"  # fallback


def _build_edges_from_relationships(
    perception_data: dict,
    object_map: dict[str, ObjectNode],
    mechanism_type: str,
) -> list[Edge]:
    """Build edges from perceived object relationships."""
    relationships = perception_data.get("relationships", [])
    edges = []

    allowed_joints = MECHANISM_JOINTS.get(mechanism_type, MECHANISM_JOINTS["rigid_body"])

    for rel in relationships:
        src = rel.get("source")
        tgt = rel.get("target")
        joint = rel.get("joint", "contact")

        if src not in object_map or tgt not in object_map:
            continue

        # Map relationship type to joint type
        joint_map = {
            "fixed": JointType.FIXED,
            "revolute": JointType.REVOLUTE,
            "prismatic": JointType.PRISMATIC,
            "belt": JointType.BELT,
            "contact": JointType.CONTACT,
        }
        joint_type = joint_map.get(joint.lower(), JointType.CONTACT)

        if joint_type not in allowed_joints:
            joint_type = JointType.CONTACT

        edges.append(Edge(
            source_id=src,
            target_id=tgt,
            joint_type=joint_type,
            contact_prob=rel.get("confidence", 0.8),
        ))

    # Default edges if no relationships detected
    if not edges and len(object_map) >= 2:
        obj_ids = list(object_map.keys())
        edges.append(Edge(
            source_id=obj_ids[0],
            target_id=obj_ids[1],
            joint_type=JointType.CONTACT,
            contact_prob=0.7,
        ))

    return edges


def _build_from_template(
    mechanism_type: str,
    user_labels: list[str],
) -> tuple[list[ObjectNode], list[Edge]]:
    """Build a physics-grounded template for a known mechanism type."""
    physics_defaults = PHYSICS_DEFAULTS.get(mechanism_type, PHYSICS_DEFAULTS["rigid_body"])

    # Template definitions per mechanism type
    templates: dict[str, list[dict]] = {
        "belt_gantry": [
            {"id": "belt",     "label": "Belt",          "type": "belt"},
            {"id": "carriage", "label": "Carriage",       "type": "carriage"},
            {"id": "pulley_a", "label": "Pulley A",       "type": "pulley"},
            {"id": "pulley_b", "label": "Pulley B",       "type": "pulley"},
            {"id": "frame",    "label": "Frame",          "type": "frame"},
        ],
        "drone": [
            {"id": "body",        "label": "Drone Body",    "type": "rigid"},
            {"id": "motor_fl",   "label": "Motor FL",      "type": "motor"},
            {"id": "motor_fr",   "label": "Motor FR",      "type": "motor"},
            {"id": "motor_bl",   "label": "Motor BL",      "type": "motor"},
            {"id": "motor_br",   "label": "Motor BR",      "type": "motor"},
            {"id": "prop_fl",    "label": "Prop FL",       "type": "propeller"},
            {"id": "prop_fr",    "label": "Prop FR",       "type": "propeller"},
            {"id": "prop_bl",    "label": "Prop BL",       "type": "propeller"},
            {"id": "prop_br",    "label": "Prop BR",       "type": "propeller"},
        ],
        "human_motion": [
            {"id": "torso",   "label": "Torso",     "type": "torso"},
            {"id": "head",    "label": "Head",      "type": "limb"},
            {"id": "arm_l",   "label": "Left Arm",  "type": "limb"},
            {"id": "arm_r",   "label": "Right Arm", "type": "limb"},
            {"id": "leg_l",   "label": "Left Leg", "type": "limb"},
            {"id": "leg_r",   "label": "Right Leg", "type": "limb"},
            {"id": "foot_l",  "label": "Left Foot", "type": "foot"},
            {"id": "foot_r",  "label": "Right Foot","type": "foot"},
        ],
        "vehicle": [
            {"id": "chassis",   "label": "Chassis",         "type": "chassis"},
            {"id": "wheel_fl",  "label": "Front Left Wheel", "type": "wheel"},
            {"id": "wheel_fr",  "label": "Front Right Wheel","type": "wheel"},
            {"id": "wheel_bl",  "label": "Back Left Wheel",  "type": "wheel"},
            {"id": "wheel_br",  "label": "Back Right Wheel", "type": "wheel"},
            {"id": "susp_fl",  "label": "FL Suspension",    "type": "suspension"},
            {"id": "susp_fr",  "label": "FR Suspension",    "type": "suspension"},
            {"id": "susp_bl",  "label": "BL Suspension",    "type": "suspension"},
            {"id": "susp_br",  "label": "BR Suspension",    "type": "suspension"},
            {"id": "motor",     "label": "Motor",            "type": "motor"},
        ],
        "robot_arm": [
            {"id": "base",     "label": "Base",          "type": "base"},
            {"id": "joint_1", "label": "Joint 1",     "type": "joint"},
            {"id": "link_1",  "label": "Link 1",      "type": "link"},
            {"id": "joint_2", "label": "Joint 2",     "type": "joint"},
            {"id": "link_2",  "label": "Link 2",      "type": "link"},
            {"id": "joint_3", "label": "Joint 3",     "type": "joint"},
            {"id": "link_3",  "label": "Link 3",      "type": "link"},
            {"id": "end_eff", "label": "End Effector",  "type": "end_effector"},
        ],
        "linkage": [
            {"id": "ground",  "label": "Ground",   "type": "ground"},
            {"id": "link_1",  "label": "Link 1",  "type": "link"},
            {"id": "link_2",  "label": "Link 2",  "type": "link"},
            {"id": "link_3",  "label": "Link 3",  "type": "link"},
            {"id": "pivot_1","label": "Pivot 1", "type": "joint"},
            {"id": "pivot_2","label": "Pivot 2", "type": "joint"},
        ],
        "pendulum": [
            {"id": "pivot",  "label": "Pivot",   "type": "joint"},
            {"id": "rod",    "label": "Rod",     "type": "rod"},
            {"id": "bob",    "label": "Bob",     "type": "bob"},
        ],
        "rigid_body": [
            {"id": "body", "label": "Body", "type": "rigid"},
        ],
        "custom": [
            {"id": "object_1", "label": user_labels[0] if len(user_labels) > 0 else "Object 1", "type": "rigid"},
            {"id": "object_2", "label": user_labels[1] if len(user_labels) > 1 else "Object 2", "type": "rigid"},
        ],
    }

    template = templates.get(mechanism_type, templates["rigid_body"])
    object_nodes: list[ObjectNode] = []
    object_map: dict[str, ObjectNode] = {}

    for i, t in enumerate(template):
        phys = physics_defaults.get(t["type"], physics_defaults.get("rigid", {"mass_kg": 1.0, "friction": 0.3, "damping": 0.1}))

        editable = {}
        for key, val in phys.items():
            if key == "mass_kg":
                editable["mass_kg"] = {"min": 0.01, "max": 500.0, "value": val}
            elif key == "friction":
                editable["friction"] = {"min": 0.0, "max": 1.0, "value": val}
            elif key == "damping":
                editable["damping"] = {"min": 0.0, "max": 2.0, "value": val}
            elif key == "tension":
                editable["belt_tension"] = {"min": 0.0, "max": 500.0, "value": val}
            elif key == "stiffness":
                editable["stiffness"] = {"min": 1.0, "max": 5000.0, "value": val}
            elif key == "torque_Nm":
                editable["torque_Nm"] = {"min": 0.0, "max": 50.0, "value": val}
            elif key == "thrust_N_kgf":
                editable["thrust_N"] = {"min": 0.0, "max": 100.0, "value": val * 9.81}

        node = ObjectNode(
            id=t["id"],
            label=t.get("label", t["id"]),
            object_type=t["type"],
            keypoints={"canonical": [[0, 0, 0]], "current": [[0, 0, 0]]},
            physics=phys,
            editable_params=editable,
        )
        object_nodes.append(node)
        object_map[t["id"]] = node

    # Build edges based on mechanism type
    edges = _build_template_edges(mechanism_type, object_map)
    return object_nodes, edges


def _build_template_edges(mechanism_type: str, object_map: dict[str, ObjectNode]) -> list[Edge]:
    """Build edges for a mechanism template."""
    edges = []

    if mechanism_type == "belt_gantry":
        edges = [
            Edge(source_id="frame",    target_id="pulley_a", joint_type=JointType.FIXED,     contact_prob=1.0),
            Edge(source_id="frame",    target_id="pulley_b", joint_type=JointType.FIXED,     contact_prob=1.0),
            Edge(source_id="belt",     target_id="carriage", joint_type=JointType.BELT,      contact_prob=0.9, belt_tension=20.0),
            Edge(source_id="belt",     target_id="pulley_a", joint_type=JointType.BELT,      contact_prob=0.95),
            Edge(source_id="belt",     target_id="pulley_b", joint_type=JointType.BELT,      contact_prob=0.95),
            Edge(source_id="carriage", target_id="frame",    joint_type=JointType.PRISMATIC, contact_prob=0.8, joint_axis=[1, 0, 0]),
        ]
    elif mechanism_type == "drone":
        motors = ["motor_fl", "motor_fr", "motor_bl", "motor_br"]
        props  = ["prop_fl", "prop_fr", "prop_bl", "prop_br"]
        for m, p in zip(motors, props):
            if m in object_map and p in object_map:
                edges.append(Edge(source_id="body", target_id=m,   joint_type=JointType.FIXED, contact_prob=1.0))
                edges.append(Edge(source_id=m,    target_id=p,   joint_type=JointType.REVOLUTE, contact_prob=1.0))
    elif mechanism_type == "vehicle":
        wheels = ["wheel_fl", "wheel_fr", "wheel_bl", "wheel_br"]
        susps  = ["susp_fl",  "susp_fr",  "susp_bl",  "susp_br"]
        for w, s in zip(wheels, susps):
            if s in object_map and w in object_map:
                edges.append(Edge(source_id="chassis", target_id=s,     joint_type=JointType.PRISMATIC, contact_prob=0.9))
                edges.append(Edge(source_id=s,        target_id=w,     joint_type=JointType.REVOLUTE,  contact_prob=0.9))
        if "motor" in object_map:
            edges.append(Edge(source_id="chassis", target_id="motor", joint_type=JointType.FIXED, contact_prob=1.0))
    elif mechanism_type == "human_motion":
        parts = [("torso", "head"), ("torso", "arm_l"), ("torso", "arm_r"),
                 ("torso", "leg_l"), ("torso", "leg_r"), ("leg_l", "foot_l"), ("leg_r", "foot_r")]
        for src, tgt in parts:
            if src in object_map and tgt in object_map:
                edges.append(Edge(source_id=src, target_id=tgt, joint_type=JointType.REVOLUTE, contact_prob=0.85))
    elif mechanism_type == "robot_arm":
        for i in range(1, 4):
            lnk, jnt = f"link_{i}", f"joint_{i}"
            prev = f"link_{i-1}" if i > 1 else "base"
            if all(k in object_map for k in [prev, lnk, jnt]):
                edges.append(Edge(source_id=prev, target_id=jnt,  joint_type=JointType.FIXED,     contact_prob=1.0))
                edges.append(Edge(source_id=jnt,  target_id=lnk, joint_type=JointType.REVOLUTE,  contact_prob=1.0))
        if "link_3" in object_map and "end_eff" in object_map:
            edges.append(Edge(source_id="link_3", target_id="end_eff", joint_type=JointType.FIXED, contact_prob=1.0))
    elif mechanism_type == "linkage":
        for i in range(1, 4):
            lnk, pvt = f"link_{i}", f"pivot_{i}"
            prev = f"link_{i-1}" if i > 1 else "ground"
            if all(k in object_map for k in [prev, lnk]):
                edges.append(Edge(source_id=prev, target_id=lnk, joint_type=JointType.REVOLUTE, contact_prob=0.9))
    elif mechanism_type == "pendulum":
        edges = [
            Edge(source_id="pivot", target_id="rod",  joint_type=JointType.FIXED,     contact_prob=1.0),
            Edge(source_id="rod",   target_id="bob",  joint_type=JointType.FIXED,     contact_prob=1.0),
        ]
    elif mechanism_type == "rigid_body":
        # No edges needed for a single rigid body
        pass
    elif mechanism_type == "custom":
        obj_ids = list(object_map.keys())
        for i in range(len(obj_ids) - 1):
            edges.append(Edge(source_id=obj_ids[i], target_id=obj_ids[i+1], joint_type=JointType.CONTACT, contact_prob=0.7))

    return edges


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_scene_graph(scene: ROCGPA_SceneGraph, session_id: str) -> Path:
    sg_dir = DATA_DIR / "sessions" / session_id / "scene_graph"
    sg_dir.mkdir(parents=True, exist_ok=True)
    path = sg_dir / "scene_graph.json"
    with open(path, "w") as f:
        f.write(scene.model_dump_json(indent=2))
    return path


def load_scene_graph(session_id: str) -> ROCGPA_SceneGraph | None:
    path = DATA_DIR / "sessions" / session_id / "scene_graph" / "scene_graph.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return ROCGPA_SceneGraph(**data)
