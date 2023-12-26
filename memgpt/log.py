import os.path
import logging
from memgpt.constants import LOGGER_NAME

# load the logger for global use
logger = logging.getLogger(LOGGER_NAME)


def reload_logger():
    global logger  # This is required to modify the global 'logger'
    # Reconfigure logger
    logger = logging.getLogger(LOGGER_NAME)


def fix_file_path(path):
    """
    Converts backslashes to forward slashes in a file path.
    This is useful for ensuring compatibility in file paths across different systems.
    """
    return path.replace("\\", "/")
