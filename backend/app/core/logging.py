"""Logging configuration."""
import logging
import sys

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"


def configure(level: int = logging.INFO):
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)])
    for name in ["uvicorn", "fastapi", "httpx", "torch"]:
        logging.getLogger(name).setLevel(logging.WARNING)
