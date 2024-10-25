import logging
import os
import threading
import time

from fastapi.testclient import TestClient
import pytest

from tests.helpers import mock_llm


def pytest_configure(config):
    logging.basicConfig(level=logging.DEBUG)


def pytest_addoption(parser):
    parser.addoption(
        "--llm-api",
        action="store",
        default="openai",
        help="backend options: openai or mock",
        choices=("openai", "mock"),
    )


@pytest.fixture(scope="module")
def llmopt(request):
    return request.config.getoption("--llm-api")


@pytest.fixture(scope="module")
def mock_llm_client(llmopt):
    if llmopt == "mock":
        print("Starting mock llm api server thread")
        print(__name__)
        thread = threading.Thread(target=mock_llm.start_mock_llm_server, daemon=True)
        thread.start()
        time.sleep(5)

        mock_llm_client = TestClient(mock_llm.app)
        yield mock_llm_client
    else:
        yield None

    # Cleanup ssl cert override
    if llmopt == "mock":
        del os.environ["REQUESTS_CA_BUNDLE"]
        os.remove(mock_llm.DEFAULT_MOCK_LLM_SSL_CERT_PATH)
        os.rmdir(mock_llm.DEFAULT_MOCK_LLM_SSL_CERT_PATH.split("/")[0])