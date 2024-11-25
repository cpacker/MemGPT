from pathlib import Path
import psutil
import logging
import subprocess

from letta.settings import settings
from letta.server.server import logger as server_logger

from installable_logger import get_logger

installable_logger = get_logger(__name__)


class LogViewer:
    log_dir: "Path"

    def __init__(self):
        self.log_dir = settings.letta_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.add_file_handler()

    def add_file_handler(self):
        """adds the rotating file handler to all monitored loggers"""
        # create a rotating file handler with 1mb size and no backup count
        file_handler = logging.handlers.RotatingFileHandler(
            maxBytes=10000, backupCount=2, filename=self.log_dir / "letta.log"
        )
        root_logger = logging.getLogger()
        uvicorn_loggers = [
            logging.getLogger(name)
            for name in logging.root.manager.loggerDict.keys()
            if name.startswith("uvicorn.")
        ]
        for logger in (installable_logger, root_logger, server_logger, *uvicorn_loggers):
            logger.addHandler(file_handler)


    def start_log_terminal(self) -> None:
        """Start the log terminal"""
        # MacOS only TODO win/linux
        log_terminal_path = Path(__file__).parent / "macOS" / "log_terminal.command"
        self.log_terminal = subprocess.Popen(
            ["open", str(log_terminal_path.absolute())], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return self.log_terminal

    def stop_log_terminal(self) -> None:
        """Stop the log terminal"""
        installable_logger.info("Stopping log terminal...")
        for process in psutil.process_iter():
            try:
                last_command = process.cmdline()
                installable_logger.info("Checking process %s", last_command)
            except (psutil.AccessDenied, psutil.NoSuchProcess,) as e:
                installable_logger.info("Error checking process %s: %s", process, e)
                continue
            if "log_terminal.command" in last_command:
                installable_logger.info("Killing log terminal process %s", process.pid)
                try:
                    process.kill()
                    installable_logger.info("Log terminal process killed")
                except Exception as e:
                    installable_logger.error("Error killing log terminal process: %s", e)
                    raise e
                return