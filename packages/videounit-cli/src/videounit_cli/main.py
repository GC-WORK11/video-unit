"""VideoUnit CLI - Main entry point."""

import typer
from rich.console import Console

from videounit_cli.__version__ import __version__
from videounit_cli.commands.init import init
from videounit_cli.commands.run import run
from videounit_cli.commands.evaluate import evaluate
from videounit_cli.commands.report import app as report_app
from videounit_cli.commands.compare import app as compare_app
from videounit_cli.commands.generate_contract import app as generate_contract_app
from videounit_cli.commands.serve import app as serve_app

app = typer.Typer(
    name="videounit",
    help="VideoUnit - AI video testing framework",
    add_completion=False,
)

console = Console()


@app.callback()
def callback():
    """VideoUnit - AI-generated videos should pass tests."""
    pass


@app.command()
def version():
    """Show VideoUnit version."""
    console.print(f"VideoUnit v{__version__}")


# Register function-based commands
app.command(name="init")(init)
app.command(name="run")(run)
app.command(name="evaluate")(evaluate)

# Register typer-app-based commands (without name to add at root level)
app.add_typer(report_app)
app.add_typer(compare_app)
app.add_typer(generate_contract_app)
app.add_typer(serve_app)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
