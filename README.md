# VideoUnit

**AI-generated videos should not just look good. They should pass tests.**

VideoUnit is an open-source framework for testing AI-generated videos. It turns video prompts into executable temporal tests, finds exact frame-level failures, scores model behavior, and generates reproducible reports.

## What It Does

```bash
videounit run tests/red_ball.yaml --video output.mp4
```

```
VideoUnit v0.1
Running 8 assertions on output.mp4

✓ object_exists:red_ball
✗ object_color_constant:red_ball
✗ object_exists:glass_cup
✓ no_random_scene_cut
✗ physical_plausibility:gravity

Score: 61.4 / 100
Report: runs/2026-05-12/report.html
```

## Core Idea

A prompt is not enough. A video needs a **contract**:

```yaml
assertions:
  - type: object_exists
    object: "red ball"
    from: 0s
    to: 6s
  - type: object_color_constant
    object: "red ball"
    expected_color: "red"
  - type: motion_direction
    object: "red ball"
    phase:
      from: 0s
      to: 3s
      expected: "mostly horizontal"
```

VideoUnit runs the contract against your video and tells you **exactly** what failed and where.

## Architecture

VideoUnit CLI/SDK are **clients** to the existing AETHER FastAPI backend. The backend runs perception models (SAM2, CoTracker3, MiDaS) and VideoUnit runs evaluators on top.

```
CLI/SDK (videounit) → HTTP → AETHER Backend (FastAPI)
                                        ↓
                              SAM2 + CoTracker3 + MiDaS
                                        ↓
                              Perception JSON + Tracks
                                        ↓
                              VideoUnit Evaluators
                                        ↓
                              Failure Report
```

## Packages

| Package | Description |
|---------|-------------|
| `packages/videounit-core/` | Core SDK: data models, VideoUnitClient |
| `packages/videounit-cli/` | CLI tool with Typer |
| `packages/videounit-evaluators/` | Evaluator plugins |
| `packages/videounit-report/` | HTML/JSON report generation |

## Quick Start

```bash
# Install
pip install -e packages/videounit-cli

# Start backend (from backend/ directory)
cd backend && uvicorn app.main:app --port 8000

# Run evaluation
videounit init
videounit run tests/red_ball.yaml --video output.mp4

# Generate report
videounit report runs/latest
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `videounit init` | Initialize project with test templates |
| `videounit run VIDEO --contract CONTRACT` | Run evaluation |
| `videounit report RUN_ID` | Generate HTML report |
| `videounit compare RUN_A RUN_B` | Compare two runs |
| `videounit generate-contract --prompt "..."` | Generate contract from prompt |
| `videounit serve` | Start backend server |

## Evaluators

| Evaluator | Checks |
|-----------|--------|
| `object_exists` | Object present throughout video |
| `object_color_constant` | Object color stays consistent |
| `object_persistence` | Object doesn't duplicate or teleport |
| `motion_direction` | Object moves in expected direction |
| `no_temporal_flicker` | No sudden frame changes |
| `no_random_scene_cut` | No unexpected scene cuts |
| `physical_plausibility` | Basic physics sanity |
| `vlm_question` | VLM-based semantic questions |

## License

Apache 2.0
