import pytest


def pytest_configure(config):
    """Configure pytest-asyncio to run all async tests automatically."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (handled by pytest-asyncio)"
    )


# Run all async tests without requiring the @pytest.mark.asyncio decorator
# on every test. Set this per-file by adding asyncio_mode = "auto" to
# pyproject.toml / pytest.ini, or globally via the hook below.
def pytest_collection_modifyitems(items):
    pytest_asyncio_tests = (item for item in items if isinstance(item, pytest.Function))
    session_scope_marker = pytest.mark.asyncio
    for async_test in pytest_asyncio_tests:
        import asyncio, inspect
        if inspect.iscoroutinefunction(async_test.function):
            async_test.add_marker(session_scope_marker, append=False)
