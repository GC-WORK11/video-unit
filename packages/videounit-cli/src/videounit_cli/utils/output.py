"""Rich console output helpers for VideoUnit CLI."""

from typing import Any

from rich.console import Console as RichConsole
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
import yaml


console = RichConsole()


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_header(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("[cyan]" + "=" * len(title) + "[/cyan]")


def print_result(result: dict[str, Any]) -> None:
    """Print evaluation results in a formatted table.

    Args:
        result: Evaluation result dictionary containing scores and failures.
    """
    overall = result.get("overall_score", result.get("overall", 0))
    categories = result.get("categories", {})

    score_color = "green" if overall >= 70 else "yellow" if overall >= 40 else "red"

    console.print()
    console.print(Panel(
        f"[bold]Overall Score:[/bold] [{score_color}]{overall:.1f}[/{score_color}]",
        title="VideoUnit Results",
        border_style="cyan",
    ))

    if categories:
        table = Table(title="Category Scores", show_header=True, header_style="bold cyan")
        table.add_column("Category", style="cyan")
        table.add_column("Score", justify="right", style="green")

        for cat, score in categories.items():
            cat_display = cat.replace("_", " ").title()
            score_val = float(score) if isinstance(score, (int, float)) else 0
            table.add_row(cat_display, f"{score_val:.1f}")

        console.print(table)

    failures = result.get("failures", [])
    if failures:
        console.print()
        failure_count = len(failures)
        critical = sum(1 for f in failures if f.get("severity") == "critical")
        print_error(f"{failure_count} failures found" + (f" ({critical} critical)" if critical else ""))

        for failure in failures[:5]:
            timestamp = failure.get("timestamp", "N/A")
            message = failure.get("message", failure.get("description", "Unknown failure"))
            severity = failure.get("severity", "fail")
            severity_color = "red" if severity in ("fail", "critical") else "yellow"
            console.print(f"  [{severity_color}]{timestamp}[/{severity_color}] - {message}")
    else:
        print_success("All assertions passed!")


def print_json(data: dict[str, Any], verbose: bool = False) -> None:
    """Print data as formatted JSON.

    Args:
        data: Dictionary to print as JSON.
        verbose: If True, use full YAML formatting; otherwise use compact JSON.
    """
    if verbose:
        console.print(Syntax(yaml.dump(data, default_flow_style=False), "yaml"))
    else:
        import json
        console.print(Syntax(json.dumps(data, indent=2), "json"))


def print_report_link(report_path: str) -> None:
    """Print a report generation success message with path."""
    console.print()
    print_success(f"Report generated: [link file://{report_path}]{report_path}[/link]")


def create_progress() -> Progress:
    """Create a Rich progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )
