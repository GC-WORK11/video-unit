# VideoUnit PRD

## Product Name

**VideoUnit**

## One-Line Thesis

**AI-generated videos should not just look good. They should pass tests.**

## Product Category

Open-source AI video evaluation, debugging, regression testing, and repair infrastructure.

## Core Idea

VideoUnit is a developer-native framework that turns video prompts into executable temporal tests. It analyzes generated videos frame-by-frame, finds exact failures, scores model behavior, creates reproducible reports, and optionally repairs broken video segments.

It is not another video generator. It is the missing infrastructure layer for AI video models.

---

# 1. Why This Product Should Exist

AI video generation is becoming powerful, but the current workflow is broken.

A user writes a prompt, generates a video, watches it manually, and guesses whether it is good. If something fails, they regenerate blindly. There is no standard way to test whether the video obeyed the prompt, maintained character identity, followed physics, preserved objects, avoided flicker, or respected temporal continuity.

For software engineering, we have unit tests, CI, debuggers, logs, assertions, regression tests, and benchmarks.

For AI video, we mostly have vibes.

VideoUnit fixes this.

The product creates a new layer:

```
Prompt → Video Model → Generated Video → VideoUnit Tests → Failure Report → Repair Loop
```

The final goal is to make AI video development measurable, reproducible, inspectable, and debuggable.

---

# 2. The Breakthrough Framing

The breakthrough is not "generate longer videos."

The breakthrough is:

> **Executable video expectations.**

A prompt is not enough. A video needs a contract.

Example:

```yaml
test_name: red_ball_falls_into_glass
prompt: "A red ball rolls off a wooden table and falls into a clear glass cup."
duration: 6s
assertions:
  - object_exists: "red ball"
    from: 0s
    to: 6s
  - object_exists: "glass cup"
    from: 0s
    to: 6s
  - object_color_constant:
      object: "red ball"
      color: "red"
  - object_motion:
      object: "red ball"
      expected: "rolls right, then falls downward"
  - no_object_duplication: "red ball"
  - no_object_disappearance: "glass cup"
  - physics_plausible: "gravity direction remains downward"
  - no_random_scene_cut
```

VideoUnit runs this contract against the generated video and outputs:

```
FAILED

Frame 47: red ball changes to orange.
Frame 83: glass cup disappears.
Frame 112: ball teleports upward after falling.

Subject consistency: 62/100
Object permanence: 41/100
Physics plausibility: 38/100
Prompt adherence: 74/100
Overall score: 54/100

Suggested repair:
Regenerate frames 72–130 with locked object masks for ball and glass.
```

This is the core product.

---

# 3. Product Vision

VideoUnit becomes the default open-source framework for testing AI-generated video.

Long-term, every serious AI video model release should be able to say:

```
VideoUnit score: 82.4
Object permanence: 88.1
Human fidelity: 79.3
Temporal consistency: 84.0
Physics: 71.2
Prompt adherence: 86.5
```

And every serious creator should be able to upload a broken AI video and ask:

```
Where exactly did this video fail, and how do I fix it?
```

---

# 4. Who This Is For

## 4.1 Primary Users

### Open-source video model builders

They need a clean way to compare model checkpoints, evaluate failures, publish reproducible results, and avoid cherry-picked demos.

### AI research labs

They need regression tests, failure localization, and measurable improvement across model versions.

### AI video tool builders

They need a quality-control layer before showing generated results to users.

### Advanced creators

They need tools to detect and repair broken AI video clips without manually rewatching everything.

## 4.2 Secondary Users

### Benchmark authors

They can contribute test packs.

### YouTubers / reviewers

They can run honest model comparisons.

### Dataset curators

They can use VideoUnit to detect low-quality generated videos.

### Model fine-tuners

They can use VideoUnit scores as training or evaluation signals.

---

# 5. What Makes VideoUnit Different

Existing benchmark tools usually answer:

```
How good is this model overall?
```

VideoUnit answers:

```
What exactly failed?
Where did it fail?
Why did it fail?
Can we reproduce it?
Can we repair it?
Did the repair improve it?
```

The key difference is that VideoUnit is not only a leaderboard. It is a full debugging and regression-testing system.

---

# 6. Product Form Factor

VideoUnit ships as four products in one ecosystem.

## 6.1 CLI

For developers, researchers, and labs.

```bash
videounit run tests/red_ball.yaml --video outputs/wan_red_ball.mp4
videounit compare --model wan2.1 --model hunyuan --suite physics-mini
videounit report runs/2026-05-12-red-ball
videounit repair outputs/broken.mp4 --contract tests/red_ball.yaml
```

## 6.2 Python SDK

For labs and tool builders.

```python
from videounit import VideoContract, VideoUnitRunner

contract = VideoContract.from_yaml("tests/red_ball.yaml")
runner = VideoUnitRunner()
result = runner.evaluate(video_path="output.mp4", contract=contract)

print(result.score)
print(result.failures)
```

## 6.3 Web Dashboard

For visual debugging.

Features:
- Upload video
- Upload or generate contract
- View timeline of failures
- See frame-level evidence
- Compare multiple models
- View score breakdown
- Export HTML/PDF report
- Regenerate only broken segments

## 6.4 Public Benchmark Suites

Curated open test packs:

```
videounit-basic-100
videounit-physics-100
videounit-object-permanence-100
videounit-human-fidelity-100
videounit-camera-control-100
videounit-text-rendering-50
videounit-long-video-50
videounit-audio-video-sync-50
```

---

# 7. MVP Definition

The MVP must prove one thing:

> VideoUnit can detect AI video failures better than a human manually guessing, and it can produce useful frame-level evidence.

## 7.1 MVP Scope

The first MVP should support:

1. Video upload or local video file input
2. Prompt input
3. Auto-generated test contract from prompt
4. Manual YAML test contract
5. Frame extraction
6. Shot boundary detection
7. Object detection
8. Object tracking
9. Identity/appearance consistency check
10. Temporal flicker check
11. Basic physics/motion sanity checks
12. Prompt adherence check through a VLM judge
13. Frame-level failure report
14. HTML report export
15. CLI + Python SDK
16. Minimal local web dashboard

## 7.2 MVP Non-Goals

- Do not train a video generation model
- Do not try to beat Sora/Veo/Kling/Runway directly
- Do not build a full video editor in v1
- Do not support every possible video failure type
- Do not over-optimize the UI before the core scoring works

---

# 8. The Killer Demo

Run the same 25 prompts across multiple video models and show VideoUnit catching failures.

Example prompt pack:

```
1. A red ball rolls off a table into a glass cup.
2. A man in a blue jacket picks up a yellow umbrella and opens it.
3. A cat walks behind a sofa and comes out the other side.
4. A woman writes the word OPEN on a whiteboard and underlines it.
5. A glass falls from a shelf and shatters on the floor.
6. A person pours milk into a transparent cup without spilling.
7. A dog jumps over a fence and lands on the grass.
8. A candle flame flickers but remains on the same candle.
9. A car turns left at an intersection without changing color.
10. A camera slowly dollies toward a statue without cutting.
```

For each model:

```
Prompt → Generated video → VideoUnit failure timeline → Score → Side-by-side replay
```

Viral post title:

> I built unit tests for AI video models. Every model failed in weird ways.

---

# 9. Architecture: CLI as Backend Client

VideoUnit CLI and SDK are **clients** to the existing AETHER FastAPI backend. No perception code is imported directly.

```
┌─────────────────────────────────────────────────────┐
│                  VideoUnit CLI/SDK                  │
│          (packages/videounit-cli/)                  │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────┐
│              AETHER FastAPI Backend                 │
│            (backend/app/main.py)                     │
│                                                     │
│  /api/perception/{id}/run   → SAM2 + CoTracker3    │
│  /api/scene-graph/{id}      → Scene graph builder  │
│  /api/videounit/evaluate    → NEW: VideoUnit eval  │
│  /api/videounit/report/{id}  → NEW: HTML report    │
└─────────────────────────────────────────────────────┘
```

**Rationale:** The existing backend already runs SAM2 segmentation, CoTracker3 tracking, and MiDaS depth estimation. VideoUnit reuses these capabilities via HTTP. The backend stays unchanged except for new `/api/videounit/*` endpoints.

---

# 10. CLI Commands

## `videounit run`

```bash
videounit run tests/red_ball.yaml --video output.mp4 [--backend http://localhost:8000]
```

Runs all assertions in a contract against a video. Outputs to `runs/{run_id}/`.

## `videounit init`

```bash
videounit init [--project .]
```

Creates a VideoUnit project with test contract templates.

## `videounit report`

```bash
videounit report runs/2026-05-12-red-ball [--format html]
```

Generates HTML or JSON report from a previous run.

## `videounit compare`

```bash
videounit compare runs/model-a runs/model-b [--format html]
```

Compares two runs side-by-side with diff.

## `videounit generate-contract`

```bash
videounit generate-contract --prompt "A red ball rolls into a glass"
```

Uses VLM to generate a first-draft contract from a prompt.

---

# 11. Contract DSL Schema

See `SPEC_CONTRACT_DSL.md` for full schema.

```yaml
version: "0.1"

test:
  id: "object_permanence_red_ball_001"
  name: "Red ball falls into glass"
  category: "object_permanence"
  difficulty: "medium"

input:
  prompt: "A red ball rolls off a wooden table and falls into a clear glass cup."
  expected_duration: "6s"

objects:
  - id: ball
    name: "red ball"
    attributes:
      color: "red"
      shape: "sphere"

  - id: glass
    name: "clear glass cup"

assertions:
  - type: object_exists
    object: ball
    from: "0s"
    to: "6s"
    tolerance:
      max_missing_duration: "0.25s"

  - type: object_color_constant
    object: ball
    expected_color: "red"

  - type: motion_direction
    object: ball
    phase:
      from: "0s"
      to: "3s"
      expected: "mostly horizontal"

  - type: no_random_scene_cut

scoring:
  weights:
    object_permanence: 0.30
    motion: 0.25
    physics: 0.25
    prompt_adherence: 0.20
```

---

# 12. Evaluator Plugins

Each assertion type maps to an **Evaluator** plugin. All evaluators receive an `EvaluationContext` and return `EvaluationResult`.

```python
class Evaluator:
    name: str
    required_inputs: list[str]

    def run(self, context: EvaluationContext) -> EvaluationResult:
        ...

class EvaluationContext:
    video_path: str
    contract: VideoContract
    perception_result: PerceptionResult  # from AETHER backend
    tracks: list[ObjectTrack]

class EvaluationResult:
    passed: bool
    score: float  # 0-100
    failures: list[Failure]
    evidence: list[EvidenceFrame]
```

**MVP Evaluators:**

| Evaluator | Assertion Types | Implementation |
|-----------|---------------|----------------|
| `ObjectExistsEvaluator` | `object_exists`, `object_absent` | SAM2 + tracking |
| `ObjectPersistenceEvaluator` | `object_persistence`, `no_object_teleportation` | CoTracker3 tracks |
| `ColorConsistencyEvaluator` | `object_color_constant` | SAM2 masks + color histogram |
| `MotionDirectionEvaluator` | `object_motion_direction` | CoTracker3 trajectories |
| `TemporalFlickerEvaluator` | `no_temporal_flicker` | Frame diff + SSIM |
| `SceneCutEvaluator` | `no_random_scene_cut` | PySceneDetect |
| `PhysicsSanityEvaluator` | `physical_plausibility` | MiDaS depth + motion vectors |
| `VLMQuestionEvaluator` | `*` (all semantic questions) | Qwen2.5-VL / Gemini |

---

# 13. Scoring System

```json
{
  "overall": 72.4,
  "categories": {
    "prompt_adherence": 81.0,
    "object_permanence": 64.5,
    "temporal_stability": 77.2,
    "motion_plausibility": 58.1,
    "camera_control": 79.6
  },
  "confidence": 0.73,
  "num_failures": 6,
  "critical_failures": 2
}
```

**Failure Severity:**
- `info`: minor artifact, probably acceptable
- `warning`: noticeable issue, maybe acceptable
- `fail`: breaks prompt or continuity
- `critical`: destroys the scene meaning

---

# 14. Report Format

Every run outputs:

```
runs/{run_id}/
  run.json          # Full results JSON
  report.html       # Human-readable HTML
  failures.json     # Failure list
  scores.json       # Score breakdown
  frames/          # Evidence frames
  tracks.json       # Object tracks
  contract.yaml     # Normalized contract
```

---

# 15. Backend Endpoints (New)

Added to `backend/app/main.py`:

```
POST   /api/videounit/evaluate        # Run full evaluation
GET    /api/videounit/report/{run_id}  # Get HTML report
POST   /api/videounit/contract/generate  # Generate contract from prompt
GET    /api/videounit/run/{run_id}     # Get run status/result
```

---

# 16. Reuse from AETHER Backend

| AETHER Component | VideoUnit Usage |
|-----------------|----------------|
| `backend/app/perception/` | Object detection, segmentation, tracking via HTTP |
| `backend/app/video/loader.py` | Frame extraction |
| `backend/app/knowledge/chromadb.py` | Failure similarity search |
| `backend/app/assistant/orchestrator.py` | VLM judge (MiniMax + Gemma4) |
| `backend/app/scene_graph/` | Scene structure analysis |
| `yolov8*.pt`, `sam2_*.pt`, `mobile_sam.pt` | Model weights |

**Removed:**
- `apps/desktop/` — Electron app deleted
- AETHER-specific physics (`backend/app/physics/`)
- Frontend PRDs (`FRONTEND_PRD.md`, `aether_main.prd.md`, etc.)
- BREAKTHROUGH_MARATHON.md, phase_*.md docs

---

# 17. Phase 1 MVP Checklist

- [ ] Write this PRD
- [ ] Write SPEC_CONTRACT_DSL.md
- [ ] Write SPEC_EVALUATORS.md
- [ ] Add `/api/videounit/evaluate` endpoint to backend
- [ ] Add `/api/videounit/report/{run_id}` endpoint to backend
- [ ] Add `/api/videounit/contract/generate` endpoint to backend
- [ ] Build `packages/videounit-core/` (SDK, data models)
- [ ] Build `packages/videounit-cli/` (Typer CLI)
- [ ] Build MVP evaluators
- [ ] Build HTML report generator
- [ ] Write 10 test contracts for killer demo
- [ ] Test full pipeline on 3+ video models

---

# 18. Success Metrics

**Developer Success:**
- Time to first report: under 5 minutes
- CLI install friction: low (`pip install videounit`)
- Contract readability: high (YAML, Git-diff friendly)
- Reproducibility: same input produces same report structure

**Evaluation Success:**
- Failure localization matches human judgment in 70%+ of obvious failures
- False positive rate manageable
- Reports useful even when detectors are imperfect

**Community Success:**
- 1,000 GitHub stars
- 10 external contributors
- 3 model adapters contributed
- 100+ public benchmark videos
- Researchers cite or reference the tool

---

# 19. Final Build Mission

Build VideoUnit as the open-source testing layer for AI video.

Not another generator.

Not a toy demo.

Not a prompt wrapper.

A serious developer tool with a clean thesis:

> **If AI video is becoming software, then AI video needs tests.**

That is the product.

That is the wedge.

That is the thing worth building.
