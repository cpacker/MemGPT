import logging
from memgpt.constants import (
    LOGGER_NAME,
    LOGGER_DEFAULT_LEVEL,
)

# Create logger for MemGPT
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOGGER_DEFAULT_LEVEL)

# create console handler and set level to debug
console_handler = logging.StreamHandler()

# create formatters
console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")  # not datetime

# add formatter to console handler
console_handler.setFormatter(console_formatter)

# add ch to logger
logger.addHandler(console_handler)
