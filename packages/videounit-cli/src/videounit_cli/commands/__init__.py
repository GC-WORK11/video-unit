"""VideoUnit CLI commands."""

from videounit_cli.commands.init import init
from videounit_cli.commands.run import run
from videounit_cli.commands.evaluate import evaluate
from videounit_cli.commands.report import report
from videounit_cli.commands.compare import compare
from videounit_cli.commands.generate_contract import generate_contract
from videounit_cli.commands.serve import serve

__all__ = [
    "init",
    "run",
    "evaluate",
    "report",
    "compare",
    "generate_contract",
    "serve",
]
