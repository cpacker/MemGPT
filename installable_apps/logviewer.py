import logging
from logging.handlers import SocketHandler

from letta.server.server import logger as server_logger

class CuteLogger:

    def apply_cuteness(self):
        root_logger = logging.getLogger()
        uvicorn_loggers = [
            logging.getLogger(name)
            for name in logging.root.manager.loggerDict.keys()
            if name.startswith("uvicorn.")
        ]
        socket_handler = SocketHandler("localhost", 19996)
        for logger in (root_logger, server_logger,*uvicorn_loggers):
            logger.addHandler(socket_handler)