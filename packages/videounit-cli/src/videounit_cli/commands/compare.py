"""Compare two VideoUnit runs side-by-side."""

import json
from pathlib import Path

import httpx
import typer

from videounit_cli.utils.backend import get_backend_url
from videounit_cli.utils.output import console, print_error, print_success


app = typer.Typer(help="Compare two VideoUnit runs side-by-side")


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID or path to run directory"),
    run_b: str = typer.Argument(..., help="Second run ID or path to run directory"),
    output: str = typer.Option(
        "comparison.html",
        "--output",
        "-o",
        help="Output path for comparison HTML file",
    ),
    backend: str = typer.Option(
        None,
        "--backend",
        "-b",
        help="Backend URL",
    ),
) -> None:
    """Compare two runs and show diff in scores/failures.

    Example:
        videounit compare run_abc123 run_def456
        videounit compare ./runs/run_abc123 ./runs/run_def456 -o diff.html
    """
    backend_url = backend or get_backend_url()

    # Load data for both runs
    data_a = _load_run(run_a, backend_url)
    data_b = _load_run(run_b, backend_url)

    if data_a is None or data_b is None:
        print_error("Failed to load one or both runs")
        raise typer.Exit(1)

    # Generate comparison
    comparison = _generate_comparison(data_a, data_b)

    # Generate HTML output
    html_content = _render_html_comparison(run_a, run_b, data_a, data_b, comparison)

    output_path = Path(output)
    output_path.write_text(html_content)
    print_success(f"Comparison saved: {output_path}")

    # Print summary to console
    _print_summary(comparison)


def _load_run(run_id: str, backend_url: str) -> dict | None:
    """Load run data from local path or backend."""
    local_path = Path(run_id)

    if local_path.exists() and local_path.is_dir():
        run_json = local_path / "run.json"
        if run_json.exists():
            return json.loads(run_json.read_text())
        print_error(f"Run result not found: {run_json}")
        return None

    # Try loading from backend
    async def fetch_from_backend():
        async with httpx.AsyncClient(timeout=60.0) as client:
            url = f"{backend_url}/api/videounit/run/{run_id}"
            try:
                response = await client.get(url)
                if response.status_code == 404:
                    print_error(f"Run {run_id} not found")
                    return None
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                print_error(f"Failed to fetch run {run_id}: {e}")
                return None

    import asyncio
    return asyncio.run(fetch_from_backend())


def _generate_comparison(data_a: dict, data_b: dict) -> dict:
    """Generate comparison data between two runs."""
    # Handle both direct format and wrapped format (with 'result' key)
    def get_run_data(d):
        return d.get("result") or d

    run_a = get_run_data(data_a)
    run_b = get_run_data(data_b)

    score_a = run_a.get("overall") or 0
    score_b = run_b.get("overall") or 0

    categories_a = run_a.get("categories") or {}
    categories_b = run_b.get("categories") or {}

    all_categories = set(categories_a.keys()) | set(categories_b.keys())

    category_deltas = {}
    for cat in all_categories:
        val_a = float(categories_a.get(cat, 0))
        val_b = float(categories_b.get(cat, 0))
        category_deltas[cat] = val_b - val_a

    failures_a = run_a.get("failures") or []
    failures_b = run_b.get("failures") or []

    failure_ids_a = {f.get("id", f.get("assertion")) for f in failures_a}
    failure_ids_b = {f.get("id", f.get("assertion")) for f in failures_b}

    added_failures = failure_ids_b - failure_ids_a
    removed_failures = failure_ids_a - failure_ids_b
    common_failures = failure_ids_a & failure_ids_b

    return {
        "score_a": score_a,
        "score_b": score_b,
        "score_delta": score_b - score_a,
        "category_deltas": category_deltas,
        "failure_count_a": len(failures_a),
        "failure_count_b": len(failures_b),
        "added_failures": [f for f in failures_b if f.get("id", f.get("assertion")) in added_failures],
        "removed_failures": [f for f in failures_a if f.get("id", f.get("assertion")) in removed_failures],
        "common_failures": list(common_failures),
    }


def _render_html_comparison(run_a: str, run_b: str, data_a: dict, data_b: dict, comparison: dict) -> str:
    """Render HTML comparison page."""
    # Get inner run data for categories
    run_data_a = data_a.get("result") or data_a
    run_data_b = data_b.get("result") or data_b

    score_a = comparison["score_a"]
    score_b = comparison["score_b"]
    score_delta = comparison["score_delta"]

    score_color_a = "green" if score_a >= 70 else "yellow" if score_a >= 40 else "red"
    score_color_b = "green" if score_b >= 70 else "yellow" if score_b >= 40 else "red"
    delta_color = "green" if score_delta >= 0 else "red"
    delta_sign = "+" if score_delta >= 0 else ""

    category_rows = ""
    for cat, delta in comparison["category_deltas"].items():
        delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        cat_display = cat.replace("_", " ").title()
        cat_a = (run_data_a.get("categories") or {}).get(cat, 0)
        cat_b = (run_data_b.get("categories") or {}).get(cat, 0)
        category_rows += f"<tr><td>{cat_display}</td><td>{cat_a:.1f}</td><td>{cat_b:.1f}</td><td class='{delta_color}'>{delta_str}</td></tr>"

    if not category_rows:
        category_rows = "<tr><td colspan='4' class='no-change'>No category data</td></tr>"

    added_rows_html = ""
    if comparison["added_failures"]:
        rows = ""
        for f in comparison["added_failures"]:
            msg = f.get("message", f.get("description", "Unknown"))
            rows += f"<tr class='failure-added'><td>{msg}</td></tr>"
        added_rows_html = f"<h4 style='color: #4ade80;'>Added Failures</h4><table><tr><th>Message</th></tr>{rows}</table>"

    removed_rows_html = ""
    if comparison["removed_failures"]:
        rows = ""
        for f in comparison["removed_failures"]:
            msg = f.get("message", f.get("description", "Unknown"))
            rows += f"<tr class='failure-removed'><td>{msg}</td></tr>"
        removed_rows_html = f"<h4 style='color: #f87171;'>Removed Failures</h4><table><tr><th>Message</th></tr>{rows}</table>"

    failure_section_content = "No failure changes"
    if comparison["added_failures"] or comparison["removed_failures"]:
        failure_section_content = added_rows_html + removed_rows_html

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>VideoUnit Run Comparison</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }}
        .header {{ text-align: center; margin-bottom: 40px; }}
        .header h1 {{ color: #00d4ff; }}
        .scores {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 40px; }}
        .score-card {{ background: #16213e; padding: 30px 50px; border-radius: 12px; text-align: center; }}
        .score-card h2 {{ margin: 0 0 10px; font-size: 18px; color: #888; }}
        .score-card .score {{ font-size: 64px; font-weight: bold; margin: 0; }}
        .score-card .run-id {{ color: #666; font-size: 14px; margin-top: 10px; }}
        .delta {{ background: #16213e; padding: 20px 40px; border-radius: 12px; text-align: center; align-self: center; }}
        .delta h2 {{ margin: 0 0 5px; color: #888; font-size: 18px; }}
        .delta .value {{ font-size: 48px; font-weight: bold; margin: 0; }}
        .section {{ background: #16213e; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .section h3 {{ color: #00d4ff; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #00d4ff; }}
        .green {{ color: #4ade80; }}
        .red {{ color: #f87171; }}
        .failure-added {{ background: rgba(74, 222, 128, 0.1); }}
        .failure-removed {{ background: rgba(248, 113, 113, 0.1); }}
        .no-change {{ color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>VideoUnit Run Comparison</h1>
    </div>

    <div class="scores">
        <div class="score-card">
            <h2>Run A</h2>
            <p class="score {score_color_a}">{score_a:.1f}</p>
            <p class="run-id">{run_a}</p>
        </div>
        <div class="delta">
            <h2>Delta</h2>
            <p class="value {delta_color}">{delta_sign}{score_delta:.1f}</p>
        </div>
        <div class="score-card">
            <h2>Run B</h2>
            <p class="score {score_color_b}">{score_b:.1f}</p>
            <p class="run-id">{run_b}</p>
        </div>
    </div>

    <div class="section">
        <h3>Category Breakdown</h3>
        <table>
            <tr><th>Category</th><th>Run A</th><th>Run B</th><th>Delta</th></tr>
            {category_rows}
        </table>
    </div>

    <div class="section">
        <h3>Failures ({comparison['failure_count_a']} → {comparison['failure_count_b']})</h3>
        <p class="no-change">{failure_section_content}</p>
    </div>
</body>
</html>"""


def _print_summary(comparison: dict) -> None:
    """Print comparison summary to console."""
    console.print()
    console.print(f"[cyan]Score Delta:[/cyan] [bold]{comparison['score_delta']:+.1f}[/bold]")
    console.print(f"[cyan]Failures:[/cyan] {comparison['failure_count_a']} → {comparison['failure_count_b']}")

    if comparison["added_failures"]:
        console.print(f"[red]  +{len(comparison['added_failures'])} new failures[/red]")
    if comparison["removed_failures"]:
        console.print(f"[green]  -{len(comparison['removed_failures'])} resolved failures[/green]")


if __name__ == "__main__":
    app()
