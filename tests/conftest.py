"""Shared test fixtures for tiedloc tests."""

import pytest

from tiedloc.metrics import get_all_metrics, _reset_metric_registry


@pytest.fixture(autouse=True)
def preserve_metric_registry():
    """Save and restore the metric registry around each test to prevent pollution."""
    saved = get_all_metrics()
    yield
    _reset_metric_registry(saved)
