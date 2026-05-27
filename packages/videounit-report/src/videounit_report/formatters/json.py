"""JSON formatter for VideoUnit reports."""

import json
from typing import Any


def format_json(result: dict, contract: dict) -> str:
    """
    Format run result as structured JSON.

    Args:
        result: Test result dict from run.json
        contract: The contract/duties YAML as dict

    Returns:
        JSON string with indented output
    """
    output = {
        "run_id": result.get("run_id"),
        "overall": result.get("overall"),
        "categories": result.get("categories", {}),
        "scores": result.get("scores", {}),
        "num_failures": result.get("num_failures"),
        "failures": result.get("failures", []),
        "passed_count": result.get("passed_count", 0),
        "timestamp": result.get("timestamp"),
        "video_duration": result.get("video_duration"),
        "contract": {
            "name": contract.get("test", {}).get("name"),
            "assertions": len(contract.get("assertions", [])),
            "duties": contract.get("duties", []),
        }
    }
    return json.dumps(output, indent=2, default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """Default serializer for JSON encoding."""
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, dict, list, tuple)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
