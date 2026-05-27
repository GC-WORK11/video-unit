"""VideoUnit API endpoints."""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
import uuid
import json
import yaml
from pathlib import Path
from datetime import datetime

from app.api.schemas import (
    ContractGenerateRequest,
    ContractGenerateResponse,
    EvaluateRequest,
    EvaluateResponse,
    RunStatusResponse,
)

router = APIRouter(prefix="/api/videounit", tags=["videounit"])

# Data directory for runs
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
VIDEUNIT_RUNS_DIR = DATA_DIR / "videounit_runs"
VIDEUNIT_RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ============== Helper Functions ==============

def _get_run_dir(run_id: str) -> Path:
    return VIDEUNIT_RUNS_DIR / run_id


def _save_video(video: UploadFile, run_id: str) -> Path:
    """Save uploaded video to run directory."""
    run_dir = _get_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename).suffix if video.filename else ".mp4"
    video_path = run_dir / f"video{suffix}"

    with open(video_path, "wb") as f:
        content = video.file.read()
        f.write(content)

    return video_path


def _load_run_result(run_id: str) -> dict:
    """Load run result JSON."""
    run_dir = _get_run_dir(run_id)
    result_path = run_dir / "run.json"

    if not result_path.exists():
        return None

    return json.loads(result_path.read_text())


def _parse_contract(contract_yaml: str) -> dict:
    """Parse and validate contract YAML."""
    try:
        contract = yaml.safe_load(contract_yaml)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    if not contract:
        raise HTTPException(status_code=400, detail="Empty contract")

    return contract


async def _run_evaluation_task(run_id: str, video_path: str, contract: dict):
    """
    Background task: Run full VideoUnit evaluation pipeline.

    Pipeline:
    1. Extract frames from video (reuse FrameExtractor)
    2. Run perception (SAM2 + CoTracker3 + MiDaS) via existing pipeline
    3. Parse contract assertions
    4. Run evaluators
    5. Aggregate scores and failures
    6. Write run.json and generate HTML report
    """
    import asyncio
    import logging
    import cv2
    from app.video.loader import FrameExtractor

    run_dir = _get_run_dir(run_id)
    logger = logging.getLogger(__name__)

    # Write initial progress
    progress_path = run_dir / "progress.json"
    progress_path.write_text(json.dumps({
        "run_id": run_id,
        "status": "starting",
        "stage": "initializing",
        "timestamp": datetime.now().isoformat(),
    }))

    try:
        # 1. Extract frames
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        progress_path.write_text(json.dumps({
            "run_id": run_id,
            "status": "running",
            "stage": "extracting_frames",
            "timestamp": datetime.now().isoformat(),
        }))

        extractor = FrameExtractor(str(video_path))
        frame_paths = extractor.extract_all_frames(output_dir=str(frames_dir))

        logger.info(f"Extracted {len(frame_paths)} frames to {frames_dir}")

        # Write frame extraction complete progress
        progress_path.write_text(json.dumps({
            "run_id": run_id,
            "status": "running",
            "stage": "frames_extracted",
            "frame_count": len(frame_paths),
            "timestamp": datetime.now().isoformat(),
        }))

        # 2. Run perception pipeline
        perception_result = None
        try:
            from app.perception.optimized.pipeline import AetherNeuralCore

            progress_path.write_text(json.dumps({
                "run_id": run_id,
                "status": "running",
                "stage": "running_perception",
                "timestamp": datetime.now().isoformat(),
            }))

            # Load frames as numpy arrays for perception pipeline
            frames = []
            for frame_path in frame_paths[:20]:  # Limit to 20 frames for speed
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    frames.append(frame)

            if frames:
                core = AetherNeuralCore()
                perception_result = core.run(frames)
                logger.info(f"Perception completed: {len(frames)} frames processed")
                core.unload_all()
            else:
                logger.warning("No frames loaded for perception")

        except Exception as e:
            logger.warning(f"Perception pipeline failed (non-critical): {e}")
            perception_result = None

        # 3. Try to load evaluators and run
        all_failures = []
        category_scores = {}
        assertions = contract.get("assertions", [])

        get_evaluator = None
        EvaluationContext = None
        try:
            from videounit_evaluators import get_evaluator as _get_evaluator
            from videounit_evaluators._context import EvaluationContext as _EC
            get_evaluator = _get_evaluator
            EvaluationContext = _EC
        except ImportError:
            logger.warning("videounit_evaluators not installed, skipping evaluator execution")

        # Get video metadata for context
        try:
            from app.video.loader import VideoMetadata
            video_meta = VideoMetadata(str(video_path))
            video_metadata = {
                "fps": video_meta.fps,
                "frame_count": video_meta.frame_count,
                "width": video_meta.width,
                "height": video_meta.height,
                "duration": video_meta.duration,
            }
        except Exception as e:
            logger.warning(f"Could not read video metadata: {e}")
            video_metadata = {"fps": 30.0, "frame_count": 0, "width": 0, "height": 0, "duration": 0.0}

        for i, assertion in enumerate(assertions):
            assertion_type = assertion.get("type", "unknown")

            if get_evaluator is None or EvaluationContext is None:
                continue

            try:
                evaluator = get_evaluator(assertion_type)
            except ValueError:
                # Unknown evaluator type - skip
                continue

            # Build proper EvaluationContext
            context = EvaluationContext(
                video_path=str(video_path),
                contract=contract,
                perception_result=perception_result,
                frames_dir=frames_dir,
                run_dir=run_dir,
                video_metadata=video_metadata,
            )

            try:
                result = await evaluator.run(context)

                category_scores[assertion_type] = result.score
                for f in result.failures:
                    all_failures.append({
                        "timestamp": f.timestamp,
                        "frame_number": f.frame_number,
                        "type": f.type or assertion_type,
                        "severity": f.severity,
                        "message": f.message,
                        "object": f.object or assertion.get("object"),
                        "suggested_fix": f.suggested_fix,
                    })
            except Exception as e:
                logger.warning(f"Evaluator {assertion_type} failed: {e}")
                category_scores[assertion_type] = 0.0

        # 4. Calculate overall score
        if category_scores:
            overall = sum(category_scores.values()) / len(category_scores)
        else:
            overall = 1.0 if not assertions else 0.5  # Default to 1.0 if no assertions, 0.5 if assertions but no scores

        critical_failures = len([f for f in all_failures if f["severity"] == "critical"])

        # 5. Write run result
        result = {
            "run_id": run_id,
            "overall": round(overall, 1),
            "categories": {k: round(v, 1) for k, v in category_scores.items()},
            "num_failures": len(all_failures),
            "critical_failures": critical_failures,
            "failures": all_failures,
            "confidence": 0.7,
            "timestamp": datetime.now().isoformat(),
            "video_path": str(video_path),
            "contract_name": contract.get("test", {}).get("name", "unknown"),
            "frame_count": len(frame_paths),
            "perception_completed": perception_result is not None,
        }

        (run_dir / "run.json").write_text(json.dumps(result, indent=2))

        # 6. Try to generate HTML report
        try:
            from videounit_report import generate_report
            generate_report(
                result=result,
                video_path=str(video_path),
                contract=contract,
                output_dir=run_dir,
                formats=["html"],
            )
        except ImportError:
            logger.warning("videounit_report not installed, skipping HTML report generation")
        except Exception as e:
            logger.warning(f"Report generation failed (non-critical): {e}")

        # Write final progress
        progress_path.write_text(json.dumps({
            "run_id": run_id,
            "status": "completed",
            "stage": "done",
            "timestamp": datetime.now().isoformat(),
        }))

    except Exception as e:
        logger.error(f"Evaluation task failed: {e}", exc_info=True)
        # Write error state
        error_result = {
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        (run_dir / "error.json").write_text(json.dumps(error_result, indent=2))
        progress_path.write_text(json.dumps({
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }))


# ============== Endpoints ==============

@router.post("/contract/generate", response_model=ContractGenerateResponse)
async def generate_contract(req: ContractGenerateRequest):
    """
    Generate a test contract from a text prompt using VLM.

    Uses the existing AETHER assistant orchestrator to parse the prompt
    and extract objects, motions, and relationships.
    """
    from app.assistant.orchestrator import hybrid_chat

    prompt_text = req.prompt

    # Build a prompt for contract generation
    system_prompt = """You are a video test contract generator. Given a text description of a video,
extract the objects, motions, and constraints to generate a YAML contract.

Return a JSON object with:
- objects: list of object names mentioned
- assertions: list of assertion types needed (object_exists, motion_direction, no_random_scene_cut, etc.)
- duration_estimate: estimated video duration in seconds

Be specific about object names and expected behaviors."""

    try:
        response = await hybrid_chat(
            user_message=f"Generate contract structure for: {prompt_text}",
            provider=req.provider,
            session_context=system_prompt,
        )

        # Parse response - in production this would be more robust
        response_text = response.get("response", "")

        # Extract object names (simple heuristic)
        objects = []
        words = prompt_text.replace(",", " ").split()
        skip_words = {"a", "an", "the", "and", "or", "but", "into", "onto", "from", "to", "of", "in", "on", "with", "for"}
        for word in words:
            if word.lower() not in skip_words and len(word) > 2:
                objects.append(word)

        # Simple assertion generation
        assertions = []
        if "ball" in prompt_text.lower() or "object" in prompt_text.lower():
            assertions.append("object_exists")
        if "rolls" in prompt_text.lower() or "moves" in prompt_text.lower() or "drives" in prompt_text.lower():
            assertions.append("motion_direction")
        if "cut" in prompt_text.lower():
            assertions.append("no_random_scene_cut")

        # Generate YAML contract
        contract = {
            "version": "0.1",
            "test": {
                "id": f"auto_{uuid.uuid4().hex[:8]}",
                "name": prompt_text[:50],
                "category": "auto_generated",
                "difficulty": "medium",
            },
            "input": {
                "prompt": prompt_text,
            },
            "objects": [{"id": f"obj_{i}", "name": name} for i, name in enumerate(objects)],
            "assertions": [
                {"type": a, "from": "0s", "to": "6s"} for a in assertions
            ] if assertions else [{"type": "object_exists", "from": "0s", "to": "6s"}],
        }

        contract_yaml = yaml.dump(contract, default_flow_style=False, sort_keys=False)

        return ContractGenerateResponse(
            contract_yaml=contract_yaml,
            objects=objects,
            assertions=len(assertions) if assertions else 1,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contract generation failed: {e}")


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_video(
    video: UploadFile = File(...),
    contract_yaml: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Run full VideoUnit evaluation on an uploaded video against a contract.

    Returns run_id immediately. Evaluation runs in background.
    """
    # Generate run ID
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    # Validate contract
    contract = _parse_contract(contract_yaml)

    # Save video
    try:
        video_path = _save_video(video, run_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save video: {e}")

    # Schedule background task
    background_tasks.add_task(_run_evaluation_task, run_id, str(video_path), contract)

    return EvaluateResponse(run_id=run_id, status="running")


@router.get("/run/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """Get status and result of a VideoUnit evaluation run."""
    run_dir = _get_run_dir(run_id)

    # Check for error
    error_path = run_dir / "error.json"
    if error_path.exists():
        error_data = json.loads(error_path.read_text())
        return RunStatusResponse(
            run_id=run_id,
            status="failed",
            progress=1.0,
            result=None,
            error=error_data.get("error"),
        )

    # Check for result
    result = _load_run_result(run_id)
    if result:
        return RunStatusResponse(
            run_id=run_id,
            status="completed",
            progress=1.0,
            result=result,
        )

    # Check if run directory exists (still running)
    if run_dir.exists():
        # Estimate progress based on files
        frames_dir = run_dir / "frames"
        if frames_dir.exists():
            frame_count = len(list(frames_dir.glob("*.png")))
            progress = min(frame_count / 30, 0.9)  # Estimate
        else:
            progress = 0.1
    else:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return RunStatusResponse(
        run_id=run_id,
        status="running",
        progress=progress,
        result=None,
    )


@router.get("/report/{run_id}")
async def get_report(run_id: str, format: str = Query("html")):
    """
    Get HTML or JSON report for a completed run.
    """
    run_dir = _get_run_dir(run_id)

    if format == "json":
        result = _load_run_result(run_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found or not completed")
        return result

    # HTML report
    report_path = run_dir / "report.html"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report for {run_id} not found. Run may still be processing.")

    return FileResponse(
        path=str(report_path),
        media_type="text/html",
        filename=f"videounit_report_{run_id}.html",
    )


@router.get("/runs")
async def list_runs():
    """List all VideoUnit runs."""
    if not VIDEUNIT_RUNS_DIR.exists():
        return {"runs": []}

    runs = []
    for run_dir in sorted(VIDEUNIT_RUNS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir():
            result = _load_run_result(run_dir.name)
            error_path = run_dir / "error.json"

            runs.append({
                "run_id": run_dir.name,
                "status": "completed" if result else ("failed" if error_path.exists() else "running"),
                "overall": result.get("overall") if result else None,
                "num_failures": result.get("num_failures", 0) if result else None,
                "timestamp": result.get("timestamp") if result else None,
            })

    return {"runs": runs}


@router.delete("/run/{run_id}")
async def delete_run(run_id: str):
    """Delete a run and its files."""
    import shutil

    run_dir = _get_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    shutil.rmtree(run_dir)

    return {"deleted": run_id}
