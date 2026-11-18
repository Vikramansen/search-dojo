"""Quiet logs and one shared archive."""

from __future__ import annotations

import pytest
from loguru import logger

from search_dojo.archive import Archive, build_archive

logger.disable("search_dojo")


@pytest.fixture(scope="session")
def archive() -> Archive:
    return build_archive()
