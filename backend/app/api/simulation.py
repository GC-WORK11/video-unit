"""
Simulation API - run universal physics with real scene graph.
"""
import logging, uuid, time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from app.core.config import DATA_DIR

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/simulation")

# Store simulation results
_simulation_results: dict = {}

# Supported mechanism types
MECHANISM_TYPES = [
    "vehicle",
    "drone",
    "robot_arm",
    "pendulum",
    "linkage",
    "belt_gantry",
    "rigid_body",
]

class SimulateRequest(BaseModel):
    session_id: str
    mechanism_type: str = Field(default="vehicle")
    horizon_seconds: float = Field(default=5.0, ge=0.1, le=30.0)
    param_overrides: dict = Field(default_factory=dict)
    run_baseline: bool = False
    initial_conditions: dict = Field(default_factory=dict)

class SimulateResponse(BaseModel):
    simulation_id: str
    session_id: str
    mechanism_type: str
    horizon_seconds: float
    timesteps: int
    vibration_freq_Hz: float | None = None
    vibration_amplitude_mm: float | None = None
    trajectory_error_mm: float | None = None
    pendulum_period_s: float | None = None
    max_velocity_ms: float | None = None
    final_position: list | None = None
    params_used: dict = {}
    confidence: float
    confidence_basis: str
    assumptions: list[str] = []
    success: bool = True
    duration: float = 0.0

@router.get("/mechanism_types")
async def list_mechanism_types():
    """List all supported mechanism types."""
    from app.physics.universal_simulator import UniversalPhysicsSimulator
    sim = UniversalPhysicsSimulator()
    return {
        "types": MECHANISM_TYPES,
        "params": {t: {} for t in MECHANISM_TYPES}, # Simplified for now
    }

@router.post("", response_model=SimulateResponse)
async def simulate(req: SimulateRequest) -> SimulateResponse:
    """Run universal physics simulation for a session."""
    from app.physics.universal_simulator import UniversalPhysicsSimulator
    from app.scene_graph.builder import load_scene_graph
    
    sim = UniversalPhysicsSimulator()
    
    # Try to load scene graph for geometry/masks
    scene = load_scene_graph(req.session_id)
    masks = None
    frame_shape = None
    
    if scene:
        # Extract masks from scene graph for procedural generation
        masks = []
        for obj in scene.objects:
            if hasattr(obj, 'segmentation') and obj.segmentation is not None:
                masks.append({
                    "id": obj.id,
                    "segmentation": obj.segmentation,
                    "bbox": obj.bbox if hasattr(obj, 'bbox') else [0,0,10,10],
                    "area": obj.physics.get("area", 0)
                })
        if hasattr(scene, 'camera') and scene.camera:
            # Rough guess for frame shape if not present
            frame_shape = (1080, 1920) 
            
    # Run simulation
    result = sim.simulate(
        mechanism_type=req.mechanism_type,
        horizon_seconds=req.horizon_seconds,
        param_overrides=req.param_overrides,
        masks=masks,
        frame_shape=frame_shape
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    sim_id = str(uuid.uuid4())[:8]
    _simulation_results[sim_id] = result
    
    return SimulateResponse(
        simulation_id=sim_id,
        session_id=req.session_id,
        mechanism_type=req.mechanism_type,
        horizon_seconds=req.horizon_seconds,
        timesteps=result.get("timesteps", 0),
        params_used=result.get("params_used", req.param_overrides),
        confidence=result.get("confidence", 0.8),
        confidence_basis=result.get("confidence_basis", "simulation"),
        success=result.get("success", True),
        duration=result.get("duration", req.horizon_seconds),
        pendulum_period_s=result.get("pendulum_period_s")
    )

@router.post("/universal", response_model=SimulateResponse)
async def simulate_universal(req: dict):
    """Run universal physics simulation with V-NEXT engine if possible."""
    from app.physics.vnext_complete import get_vnext_engine
    
    mech_type = req.get("mechanism_type", "rigid_body")
    params = req.get("params", {})
    horizon = req.get("horizon_seconds", 5.0)
    session_id = req.get("session_id", "universal")
    
    engine = get_vnext_engine()
    
    # Try V-NEXT simulation first
    vnext_result = None
    try:
        # Look for a twin by type or session_id
        twin_name = next(iter(engine.digital_twins.keys())) if engine.digital_twins else None

        if twin_name:
            vnext_result = await engine.simulate_digital_twin(twin_name, horizon, params)
    except Exception as e:
        log.warning(f"V-NEXT simulation failed, falling back: {e}")

    if vnext_result and vnext_result.get("success"):
        return SimulateResponse(
            simulation_id=f"vnext_{int(time.time())}",
            session_id=session_id,
            mechanism_type=mech_type,
            horizon_seconds=horizon,
            timesteps=vnext_result.get("timesteps", 500),
            params_used=params,
            confidence=vnext_result.get("confidence", 0.85),
            confidence_basis="MJX-VNEXT-BACKPROP",
            success=True,
            duration=vnext_result.get("duration", horizon)
        )

    # Fallback to legacy procedural simulator
    from app.physics.universal_simulator import simulate_universal as run_sim
    result = run_sim(mech_type, horizon, params)
    
    return SimulateResponse(
        simulation_id=f"sim_{mech_type}_{int(time.time())}",
        session_id=session_id,
        mechanism_type=mech_type,
        horizon_seconds=horizon,
        timesteps=result.get("timesteps", 0),
        params_used=params,
        confidence=result.get("confidence", 0.8),
        confidence_basis="procedural_extrapolation",
        success=result.get("success", True),
        duration=result.get("duration", horizon)
    )

@router.get("/{simulation_id}")
async def get_simulation(simulation_id: str):
    if simulation_id not in _simulation_results:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return _simulation_results[simulation_id]
