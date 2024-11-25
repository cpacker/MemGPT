from logging import getLogger

def get_logger(name: str):
    logger = getLogger("installable_apps")
    logger.setLevel("DEBUG")
    return logger.getChild(name)