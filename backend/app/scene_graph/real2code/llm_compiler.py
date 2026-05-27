"""
AETHER Real2Code: LLM-Driven URDF Compiler
==========================================

This module implements the V-NEXT breakthrough: replacing heuristics with
an LLM-based physical code compiler.

Instead of 'guessing' the mechanism type, we provide the 3D orientations
and relative motions of discovered parts to an LLM (Gemma 4). The LLM
autonomously writes the URDF/MJCF XML.

Workflow:
1. SPLART Discover: Extract 3D rigid parts and their trajectories.
2. Prompt LLM: Provide part orientations and connectivity data.
3. Compile: LLM returns a structured JSON RobotSpec or URDF XML.
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional
import numpy as np
from .urdf_compiler import URDFCompiler, RobotSpec, LinkSpec, JointSpec
from ...core.ai_client import UniversalAIClient

log = logging.getLogger(__name__)


class LLMPhysicsCompiler:
    """
    Interfaces with a Cloud LLM (MiniMax/GPT-4) to compile physical simulation 
    code from motion data with General Intelligence.
    """
    
    def __init__(self, provider: str = "minimax"):
        self.base_compiler = URDFCompiler()
        self.client = UniversalAIClient(
            provider=provider,
            model=os.getenv("AI_MODEL", "MiniMax-M2.7-highspeed"),
            api_key=os.getenv("MINIMAX_API_KEY", "")
        )
        
    async def compile_from_observations(
        self,
        parts: List[Dict[str, Any]],
        trajectories: Dict[str, np.ndarray],
    ) -> str:
        """
        Main entry point: Video Data → LLM → URDF XML.
        
        Args:
            parts: List of discovered rigid parts with geometry.
            trajectories: Dict of part names to their SE(3) trajectories.
        """
        # 1. Prepare physical metadata for the LLM
        context = self._prepare_llm_context(parts, trajectories)
        
        # 2. Build the Physical Compiler Prompt
        prompt = self._build_prompt(context)
        
        # 3. Call Cloud LLM (MiniMax)
        log.info(f"Requesting URDF compilation from {self.client.provider}...")
        try:
            messages = [
                {"role": "system", "content": "You are a Physical Compiler. Output JSON ONLY."},
                {"role": "user", "content": prompt}
            ]
            response_text = await self.client.chat_completion(messages)
            
            # Extract JSON from potential markdown wrap
            clean_json = response_text.replace("```json", "").replace("```", "").strip()
            spec_data = json.loads(clean_json)
            
            # 4. Map LLM JSON to validated RobotSpec
            robot_spec = self._map_to_spec(spec_data)
            
            # 5. Final XML compilation
            return self.base_compiler.compile(robot_spec)
            
        except Exception as e:
            log.error(f"Cloud LLM Compilation failed: {e}. Falling back to default fixed-base spec.")
            return self.base_compiler.from_kinematic_tree(parts, [], "fallback_robot")

    def _prepare_llm_context(self, parts: List[Dict], trajectories: Dict) -> Dict:
        """Extract spatial relationships for LLM reasoning."""
        context = []
        for i, part in enumerate(parts):
            name = part.get("name", f"part_{i}")
            traj = trajectories.get(name, np.zeros((1, 3)))
            
            # Calculate mean motion
            motion_vector = np.ptp(traj, axis=0) if len(traj) > 1 else np.zeros(3)
            
            context.append({
                "name": name,
                "bbox": part.get("bbox", [0,0,0,1,1,1]),
                "average_motion": motion_vector.tolist(),
                "relative_to_root": traj[0].tolist() if len(traj) > 0 else [0,0,0]
            })
        return {"parts": context}

    def _build_prompt(self, context: Dict) -> str:
        """The 'Real2Code' system prompt."""
        return f"""
You are a Robotics Engineer and Physics Compiler.
Task: Analyze the discovered rigid parts of a mechanism and write a valid URDF RobotSpec in JSON format.

OBSERVED DATA:
{json.dumps(context, indent=2)}

INSTRUCTIONS:
1. Identify the most likely physical joints (revolute, prismatic, or fixed) based on part motion.
2. Determine the parent-child hierarchy.
3. Calculate the joint origin points from the relative_to_root coordinates.
4. Output a JSON object with:
   - "name": Robot name
   - "links": list of {{ "name", "mass", "geometry": "box", "size": [x,y,z] }}
   - "joints": list of {{ "name", "parent", "child", "type", "axis": [x,y,z], "origin": [x,y,z] }}

JSON OUTPUT ONLY:
"""

    def _map_to_spec(self, data: Dict) -> RobotSpec:
        """Validate and map LLM JSON to internal dataclasses."""
        links = []
        for l in data.get("links", []):
            links.append(LinkSpec(
                name=l["name"],
                mass=l.get("mass", 1.0),
                visual_size=l.get("size", [0.1, 0.1, 0.1])
            ))
            
        joints = []
        for j in data.get("joints", []):
            joints.append(JointSpec(
                name=j["name"],
                parent_link=j["parent"],
                child_link=j["child"],
                joint_type=j["type"],
                axis=np.array(j.get("axis", [0,0,1])),
                origin_xyz=np.array(j.get("origin", [0,0,0]))
            ))
            
        return RobotSpec(name=data.get("name", "aether_gen"), links=links, joints=joints)
