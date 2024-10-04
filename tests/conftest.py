import logging


def pytest_configure(config):
    logging.basicConfig(level=logging.DEBUG)
