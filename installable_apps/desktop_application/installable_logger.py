from logging import getLogger

def get_logger(name: str):
    logger = getLogger("letta_installable_apps")
    logger.setLevel("DEBUG")
    return logger.getChild(name)