import logging
from memgpt.log import logger
from memgpt.constants import LOGGER_LOG_LEVELS
import pytest


def test_log_debug():
    # test setting logging level
    assert logging.DEBUG == LOGGER_LOG_LEVELS["DEBUG"]
    logger.setLevel(LOGGER_LOG_LEVELS["DEBUG"])
    assert logger.isEnabledFor(logging.DEBUG)

    # Assert that the message was logged
    assert logger.hasHandlers()
    logger.debug("This is a Debug message")
    assert 1 == 1
