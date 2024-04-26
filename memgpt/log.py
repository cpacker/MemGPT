import logging
import os
import os.path
from logging.handlers import RotatingFileHandler

from memgpt.constants import (
    LOGGER_DEFAULT_LEVEL,
    LOGGER_DIR,
    LOGGER_FILE_BACKUP_COUNT,
    LOGGER_FILENAME,
    LOGGER_MAX_FILE_SIZE,
    LOGGER_NAME,
)

# Checking if log directory exists
if not os.path.exists(LOGGER_DIR):
    os.makedirs(LOGGER_DIR, exist_ok=True)

# Create logger for MemGPT
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(LOGGER_DEFAULT_LEVEL)

# create console handler and set level to debug
console_handler = logging.StreamHandler()

# create rotatating file handler
file_handler = RotatingFileHandler(
    os.path.join(LOGGER_DIR, LOGGER_FILENAME), maxBytes=LOGGER_MAX_FILE_SIZE, backupCount=LOGGER_FILE_BACKUP_COUNT
)

# create formatters
console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")  # not datetime
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# add formatter to console handler
console_handler.setFormatter(console_formatter)

# add formatter for file handler
file_handler.setFormatter(file_formatter)

# add ch to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
