"""Chart data generators for HTML reports."""

from typing import Any


def get_radar_chart_data(categories: dict) -> dict:
    """
    Return Recharts radar chart data structure.

    Args:
        categories: Dict of category scores e.g. {"accuracy": 85, "completeness": 90}

    Returns:
        Dict with labels and data arrays for Recharts RadarChart
    """
    label_map = {
        "accuracy": "Accuracy",
        "completeness": "Completeness",
        "coherence": "Coherence",
        "quality": "Quality",
        "timing": "Timing",
    }

    labels = []
    data = []

    for key, label in label_map.items():
        labels.append(label)
        data.append({
            "category": label,
            "score": categories.get(key, 0),
            "fullMark": 100,
        })

    return {
        "labels": labels,
        "data": data,
    }


def get_bar_chart_data(runs: list[dict]) -> dict:
    """
    Return data for score comparison bar chart.

    Args:
        runs: List of run result dicts with run_id and scores

    Returns:
        Dict with labels and data arrays for Recharts BarChart
    """
    labels = []
    accuracy_data = []
    completeness_data = []
    coherence_data = []
    quality_data = []
    timing_data = []

    for run in runs:
        run_id = run.get("run_id", f"Run {len(labels) + 1}")
        # Truncate long run IDs for display
        labels.append(run_id[:12] if len(run_id) > 12 else run_id)

        scores = run.get("scores", {})
        accuracy_data.append(scores.get("accuracy", 0))
        completeness_data.append(scores.get("completeness", 0))
        coherence_data.append(scores.get("coherence", 0))
        quality_data.append(scores.get("quality", 0))
        timing_data.append(scores.get("timing", 0))

    return {
        "labels": labels,
        "datasets": [
            {"name": "Accuracy", "data": accuracy_data},
            {"name": "Completeness", "data": completeness_data},
            {"name": "Coherence", "data": coherence_data},
            {"name": "Quality", "data": quality_data},
            {"name": "Timing", "data": timing_data},
        ],
    }


def get_score_distribution(scores: dict) -> list[dict]:
    """
    Calculate score distribution for histogram.

    Args:
        scores: Dict of category scores

    Returns:
        List of dicts with score ranges and counts
    """
    ranges = [
        ("0-20", 0, 20),
        ("21-40", 21, 40),
        ("41-60", 41, 60),
        ("61-80", 61, 80),
        ("81-100", 81, 100),
    ]

    distribution = []
    for label, low, high in ranges:
        count = sum(1 for v in scores.values() if low <= v <= high)
        distribution.append({"range": label, "count": count})

    return distribution


def get_trend_data(history: list[dict]) -> dict:
    """
    Extract trend data for line charts.

    Args:
        history: List of historical run results

    Returns:
        Dict with timestamps and score series
    """
    timestamps = []
    overall_scores = []
    accuracy_trend = []
    completeness_trend = []

    for run in history:
        timestamps.append(run.get("run_id", ""))
        overall_scores.append(run.get("overall", 0))
        scores = run.get("scores", {})
        accuracy_trend.append(scores.get("accuracy", 0))
        completeness_trend.append(scores.get("completeness", 0))

    return {
        "timestamps": timestamps,
        "series": {
            "overall": overall_scores,
            "accuracy": accuracy_trend,
            "completeness": completeness_trend,
        },
    }
