"""HTML report generation."""

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..utils.timeline import build_failure_timeline


def generate_html(
    result: dict,
    contract: dict,
    video_path: str,
    evidence_frames: list[dict],
    timeline: list[dict],
) -> str:
    """
    Generate an HTML report from test results.

    Args:
        result: Test result dict with failures and scores
        contract: The contract/duties as dict
        video_path: Path to the video file
        evidence_frames: List of extracted evidence frames
        timeline: Built failure timeline

    Returns:
        Complete HTML document as string
    """
    # Calculate overall score
    # Support both {"scores": {"total": X}} and {"overall": X, "categories": {...}} formats
    scores = result.get("scores", {})
    if scores:
        total_score = scores.get("total", 0)
    else:
        total_score = result.get("overall", result.get("score", 0))
    max_score = 100

    # Determine overall status
    if total_score >= 80:
        overall_status = "pass"
        overall_label = "Excellent"
    elif total_score >= 60:
        overall_status = "warn"
        overall_label = "Needs Work"
    else:
        overall_status = "fail"
        overall_label = "Critical Issues"

    # Count passes and failures
    failures = result.get("failures", [])
    pass_count = result.get("passed_count", 0)
    fail_count = len(failures)

    # Generate run ID from timestamp or use default
    run_id = result.get("run_id", f"run_{int(datetime.now().timestamp())}")

    # Timestamp
    timestamp = result.get("timestamp", datetime.now().isoformat())
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (ValueError, AttributeError):
            pass

    # Build score cards - use categories if scores is empty
    categories_data = result.get("categories", {})
    score_cards = _build_score_cards(categories_data if not scores else scores)

    # Build chart data
    radar_labels, radar_data = _build_radar_data(categories_data if not scores else scores)

    # Calculate timeline positions for video markers
    video_duration = result.get("video_duration", 60)  # Default 60s if unknown
    for event in timeline:
        frame = event.get("frame", 0)
        event["position"] = min((frame / 30) / video_duration * 100, 100) if video_duration > 0 else 0

    # Format contract as YAML
    contract_yaml = yaml.dump(contract, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Format raw JSON
    raw_json = json.dumps(result, indent=2, default=str)

    # Video path for display
    video_display_path = video_path if video_path else ""

    # Video filename for display
    import os
    video_filename = os.path.basename(video_path) if video_path else "unknown"

    return _render_template(
        run_id=run_id,
        timestamp=timestamp,
        overall_score=total_score,
        overall_status=overall_status,
        overall_label=overall_label,
        score_cards=score_cards,
        radar_labels=json.dumps(radar_labels),
        radar_data=json.dumps(radar_data),
        pass_count=pass_count,
        fail_count=fail_count,
        failures=failures,
        timeline=timeline,
        evidence_frames=evidence_frames,
        video_path=video_filename,
        contract=contract,
        contract_yaml=contract_yaml,
        raw_json=raw_json,
        critical_failures=result.get("critical_failures", 0),
        frame_count=result.get("frame_count", 0),
    )


def _build_score_cards(scores: dict) -> list[dict]:
    """Build score card data from scores dict."""
    # Map evaluator names to display names
    name_map = {
        "accuracy": "Accuracy",
        "completeness": "Completeness",
        "coherence": "Coherence",
        "quality": "Quality",
        "timing": "Timing",
        "object_exists": "Object Exists",
        "motion_direction": "Motion Direction",
        "no_random_scene_cut": "Scene Cuts",
        "color_constant": "Color Constant",
        "temporal_flicker": "Temporal Flicker",
        "vlm_judge": "VLM Judge",
    }
    icon_map = {
        "accuracy": "target",
        "completeness": "layers",
        "coherence": "brain",
        "quality": "star",
        "timing": "clock",
        "object_exists": "eye",
        "motion_direction": "arrow",
        "no_random_scene_cut": "film",
        "color_constant": "palette",
        "temporal_flicker": "lightning",
        "vlm_judge": "cpu",
    }

    cards = []
    for key, value in scores.items():
        title = name_map.get(key, key.replace("_", " ").title())
        icon = icon_map.get(key, "check")
        max_val = 100
        percent = min(value / max_val * 100, 100)

        if value >= 80:
            level = "pass"
        elif value >= 60:
            level = "warn"
        else:
            level = "fail"

        cards.append({
            "icon": icon,
            "title": title,
            "value": value,
            "max": max_val,
            "percent": percent,
            "level": level,
            "detail": f"{value}% threshold met" if value >= 80 else f"{100 - value}% below target",
        })

    return cards


def _build_radar_data(scores: dict) -> tuple[list[str], list[int]]:
    """Build radar chart labels and data from scores dict."""
    # If scores has generic keys, use label_map; otherwise use scores keys directly
    label_map = {
        "accuracy": "Accuracy",
        "completeness": "Completeness",
        "coherence": "Coherence",
        "quality": "Quality",
        "timing": "Timing",
        "object_exists": "Object Exists",
        "motion_direction": "Motion Direction",
        "no_random_scene_cut": "Scene Cuts",
        "color_constant": "Color Constant",
        "temporal_flicker": "Temporal Flicker",
        "vlm_judge": "VLM Judge",
    }

    # Use scores keys if they exist, otherwise use label_map keys
    if scores:
        first_key = next(iter(scores.keys()), None)
        if first_key and first_key in label_map:
            # Use predefined labels for generic scores
            labels = [label_map[k] for k in label_map if k in scores]
            data = [scores.get(k, 0) for k in label_map if k in scores]
        else:
            # Use scores keys directly for evaluator-style data
            labels = [label_map.get(k, k.replace("_", " ").title()) for k in scores.keys()]
            data = list(scores.values())
    else:
        labels = []
        data = []

    # Ensure we have exactly 5 items for the radar chart
    while len(labels) < 5:
        labels.append("Score")
        data.append(0)
    labels = labels[:5]
    data = data[:5]

    return labels, data


def _render_template(**context) -> str:
    """Load and render the Jinja2 template."""
    import jinja2

    template_path = Path(__file__).parent.parent / "templates" / "report.html"
    template_content = template_path.read_text(encoding="utf-8")

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        autoescape=True,
    )
    template = env.from_string(template_content)
    return template.render(**context)


def _process_loops(html: str, context: dict) -> str:
    """Process Jinja2 for loops in the template."""
    import re

    # Process score_cards loop
    score_cards_pattern = r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}"

    def replace_score_cards(match):
        loop_var = match.group(1)
        loop_iterable = match.group(2)
        loop_body = match.group(3)

        items = context.get(loop_iterable, [])
        result = []

        for item in items:
            if isinstance(item, dict):
                processed = loop_body
                for k, v in item.items():
                    processed = processed.replace("{{ " + loop_var + "." + k + " }}", str(v))
                result.append(processed)
            else:
                result.append(loop_body.replace("{{ " + loop_var + " }}", str(item)))

        return "".join(result)

    html = re.sub(score_cards_pattern, replace_score_cards, html, flags=re.DOTALL)

    # Process failures loop
    failures_pattern = r"\{%\s*for\s+(\w+)\s+in\s+failures\s*%\}(.*?)\{%\s*endfor\s*%\}"

    def replace_failures(match):
        loop_var = match.group(1)
        loop_body = match.group(2)
        failures = context.get("failures", [])
        result = []

        for failure in failures:
            processed = loop_body
            if isinstance(failure, dict):
                for k, v in failure.items():
                    processed = processed.replace("{{ " + loop_var + "." + k + " }}", str(v))
                # Handle nested dicts like failure.severity|default('fail')
                processed = re.sub(
                    r"\{\{\s*" + loop_var + r"\.(\w+)\|default\(['\"](.*?)['\"]\)\s*\}\}",
                    lambda m: str(failure.get(m.group(1), m.group(2))),
                    processed
                )
            result.append(processed)

        return "".join(result)

    html = re.sub(failures_pattern, replace_failures, html, flags=re.DOTALL)

    # Process timeline loop
    timeline_pattern = r"\{%\s*for\s+(\w+)\s+in\s+timeline\s*%\}(.*?)\{%\s*endfor\s*%\}"

    def replace_timeline(match):
        loop_var = match.group(1)
        loop_body = match.group(2)
        timeline = context.get("timeline", [])
        result = []

        for event in timeline:
            processed = loop_body
            if isinstance(event, dict):
                for k, v in event.items():
                    processed = processed.replace("{{ " + loop_var + "." + k + " }}", str(v))
            result.append(processed)

        return "".join(result)

    html = re.sub(timeline_pattern, replace_timeline, html, flags=re.DOTALL)

    # Process evidence_frames loop
    evidence_pattern = r"\{%\s*for\s+(\w+)\s+in\s+evidence_frames\s*%\}(.*?)\{%\s*endfor\s*%\}"

    def replace_evidence(match):
        loop_var = match.group(1)
        loop_body = match.group(2)
        frames = context.get("evidence_frames", [])
        result = []

        for frame in frames:
            processed = loop_body
            if isinstance(frame, dict):
                for k, v in frame.items():
                    processed = processed.replace("{{ " + loop_var + "." + k + " }}", str(v))
                # Handle |default filter
                processed = re.sub(
                    r"\{\{\s*" + loop_var + r"\.(\w+)\|default\(['\"](.*?)['\"]\)\s*\}\}",
                    lambda m: str(frame.get(m.group(1), m.group(2))),
                    processed
                )
            result.append(processed)

        return "".join(result)

    html = re.sub(evidence_pattern, replace_evidence, html, flags=re.DOTALL)

    # Process timeline markers loop
    markers_pattern = r"\{%\s*for\s+(\w+)\s+in\s+timeline\s*%\}(.*?)\{%\s*endfor\s*%\}"
    # Already processed above, but there may be another instance for video markers

    return html
