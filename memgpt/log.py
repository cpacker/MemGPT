import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from memgpt.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MemGPT")
logger.setLevel(logging.DEBUG if settings.debug else logging.INFO)

# create console handler and set level to debug
console_handler = logging.StreamHandler()

# create rotatating file handler
logfile = Path(settings.memgpt_dir / "logs" / "MemGPT.log")
logfile.parent.mkdir(parents=True, exist_ok=True)
logfile.touch(exist_ok=True)
file_handler = RotatingFileHandler(
    logfile,
    maxBytes=1024**2 * 10,
    backupCount=3
)

# create formatters
console_formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")  # not datetime
console_handler.setFormatter(console_formatter)

file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

# add ch to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
