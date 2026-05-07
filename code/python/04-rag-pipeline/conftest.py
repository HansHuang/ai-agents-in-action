import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests requiring external services")
