"""Initialize a new VideoUnit project."""

from pathlib import Path

import typer

from videounit_cli.utils.output import console, print_success, print_error

INIT_COMMAND = """\
videounit init - Initialize a VideoUnit project

This command creates a new VideoUnit project with:
- videounit.yaml: Project configuration file
- tests/contracts/: Directory for test contracts
- Example contract files to get started

Usage:
    videounit init [PROJECT_DIR]

Args:
    project_dir: Directory to initialize (default: current directory)

Example:
    videounit init my-video-tests
    cd my-video-tests
    videounit run tests/example_basic.yaml --video output.mp4
"""

INIT_TEMPLATE = """\
# VideoUnit Project Configuration
# https://github.com/videounit/videounit

version: "0.1"
name: "{project_name}"
description: "VideoUnit test project"

# Backend configuration
backend: "http://localhost:8000"

# Test settings
tests:
  contracts_dir: "tests/contracts"
  output_dir: "runs"

# Default evaluation settings
evaluation:
  max_frames: 16
  fail_threshold: 70
"""

EXAMPLE_BASIC_CONTRACT = """\
# Example VideoUnit Contract - Basic Object Detection
# This contract tests basic object existence and visibility

version: "0.1"

test:
  id: "example_basic_001"
  name: "Basic Object Detection Example"
  category: "basic"
  difficulty: "easy"

input:
  prompt: "A red ball rolls across a white table."
  expected_duration: "5s"

objects:
  - id: ball
    name: "red ball"
    attributes:
      color: "red"
      shape: "sphere"

  - id: table
    name: "white table"
    attributes:
      color: "white"

assertions:
  - type: object_exists
    object: ball
    from: "0s"
    to: "5s"
    tolerance:
      max_missing_duration: "0.25s"

  - type: object_exists
    object: table
    from: "0s"
    to: "5s"

  - type: object_color_constant
    object: ball
    expected_color: "red"

  - type: no_random_scene_cut

scoring:
  weights:
    object_permanence: 0.30
    color_consistency: 0.25
    temporal_stability: 0.25
    prompt_adherence: 0.20
"""

EXAMPLE_PHYSICS_CONTRACT = """\
# Example VideoUnit Contract - Physics Plausibility
# This contract tests physics-based assertions

version: "0.1"

test:
  id: "example_physics_001"
  name: "Physics Plausibility Example"
  category: "physics"
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

  - id: table
    name: "wooden table"
    attributes:
      color: "brown"
      material: "wood"

  - id: cup
    name: "clear glass cup"
    attributes:
      material: "glass"
      transparency: "clear"

assertions:
  - type: object_exists
    object: ball
    from: "0s"
    to: "6s"
    tolerance:
      max_missing_duration: "0.25s"

  - type: object_exists
    object: cup
    from: "0s"
    to: "6s"

  - type: object_color_constant
    object: ball
    expected_color: "red"

  - type: motion_direction
    object: ball
    phase:
      from: "0s"
      to: "3s"
      expected: "horizontal_right"
    then:
      from: "3s"
      to: "6s"
      expected: "vertical_downward"

  - type: no_object_teleportation
    object: ball

  - type: no_object_disappearance
    object: cup

  - type: physics_plausible
    object: ball
    gravity_direction: "downward"

  - type: no_random_scene_cut

scoring:
  weights:
    object_permanence: 0.25
    motion: 0.25
    physics: 0.25
    prompt_adherence: 0.25
"""

GITIGNORE_CONTENT = """\
# VideoUnit output
runs/
*.mp4
*.mov
*.avi

# Python
__pycache__/
*.py[cod]
*$py.class

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db
"""


def init(
    project_dir: str = typer.Argument(".", help="Directory to initialize"),
) -> None:
    """Initialize a VideoUnit project with test contract templates.

    Creates a new VideoUnit project structure with configuration
    file and example test contracts.
    """
    project_path = Path(project_dir).resolve()

    if project_path.exists() and any(project_path.iterdir()):
        console.print("[yellow]Warning:[/yellow] Directory is not empty. Files may be overwritten.")

    project_name = project_path.name

    console.print(f"[cyan]Initializing VideoUnit project:[/cyan] {project_name}")

    project_path.mkdir(parents=True, exist_ok=True)

    config_content = INIT_TEMPLATE.format(project_name=project_name)
    config_path = project_path / "videounit.yaml"
    config_path.write_text(config_content)
    print_success(f"Created: {config_path}")

    tests_dir = project_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    print_success(f"Created: {tests_dir}/")

    contracts_dir = tests_dir / "contracts"
    contracts_dir.mkdir(exist_ok=True)

    example_basic = contracts_dir / "example_basic.yaml"
    example_basic.write_text(EXAMPLE_BASIC_CONTRACT)
    print_success(f"Created: {example_basic}")

    example_physics = contracts_dir / "example_physics.yaml"
    example_physics.write_text(EXAMPLE_PHYSICS_CONTRACT)
    print_success(f"Created: {example_physics}")

    gitignore_path = project_path / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(GITIGNORE_CONTENT)
        print_success(f"Created: {gitignore_path}")

    console.print()
    console.print("[bold green]VideoUnit project initialized successfully![/bold green]")
    console.print()
    console.print("Next steps:")
    console.print(f"  [dim]cd {project_path}[/dim]")
    console.print(f"  [dim]videounit serve &[/dim]")
    console.print(f"  [dim]videounit run tests/contracts/example_basic.yaml --video your_video.mp4[/dim]")
