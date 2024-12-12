import logging

import pytest


def pytest_configure(config):
    logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_e2b_api_key_none():
    from letta.settings import tool_settings

    # Store the original value of e2b_api_key
    original_api_key = tool_settings.e2b_api_key

    # Set e2b_api_key to None
    tool_settings.e2b_api_key = None

    # Yield control to the test
    yield

    # Restore the original value of e2b_api_key
    tool_settings.e2b_api_key = original_api_key
