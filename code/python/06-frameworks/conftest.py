"""pytest configuration for code/python/06-frameworks/."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring a live OPENAI_API_KEY (skipped by default)",
    )
