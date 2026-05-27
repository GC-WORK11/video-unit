"""Main report generator for VideoUnit."""

from pathlib import Path

from .formatters.html import generate_html
from .formatters.json import format_json
from .formatters.markdown import format_markdown
from .utils.frames import extract_evidence_frames
from .utils.timeline import build_failure_timeline


def generate_report(
    result: dict,
    video_path: str,
    contract: dict,
    output_dir: Path,
    formats: list[str] = None,
) -> dict[str, Path]:
    """
    Generate reports in multiple formats.

    Args:
        result: Test result dict from run.json
        video_path: Path to the test video file
        contract: The contract/duties YAML as dict
        output_dir: Directory to write reports
        formats: List of formats to generate ("html", "json", "markdown")

    Returns:
        Dict mapping format -> output_path
        e.g. {"html": Path("runs/abc/report.html"), "json": Path("runs/abc/report.json")}
    """
    if formats is None:
        formats = ["html", "json"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract evidence frames
    evidence_frames = extract_evidence_frames(
        video_path, result.get("failures", []), output_dir / "frames"
    )

    # Build failure timeline
    timeline = build_failure_timeline(result.get("failures", []))

    outputs = {}

    if "html" in formats:
        html_path = output_dir / "report.html"
        html = generate_html(
            result=result,
            contract=contract,
            video_path=video_path,
            evidence_frames=evidence_frames,
            timeline=timeline,
        )
        html_path.write_text(html, encoding="utf-8")
        outputs["html"] = html_path

    if "json" in formats:
        json_path = output_dir / "report.json"
        json_content = format_json(result, contract)
        json_path.write_text(json_content, encoding="utf-8")
        outputs["json"] = json_path

    if "markdown" in formats:
        md_path = output_dir / "summary.md"
        md_content = format_markdown(result, contract)
        md_path.write_text(md_content, encoding="utf-8")
        outputs["markdown"] = md_path

    return outputs
