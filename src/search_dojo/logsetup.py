"""Keep training logs out of the belt output unless you ask."""

from __future__ import annotations

import os
import sys

from loguru import logger


def configure_logging() -> None:
    """Send loguru to stderr at ``SEARCH_DOJO_LOG`` (default WARNING)."""
    level = os.environ.get("SEARCH_DOJO_LOG", "WARNING")
    logger.remove()
    logger.add(sys.stderr, level=level)
