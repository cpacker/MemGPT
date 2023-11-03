import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pexpect"])
import pexpect

from .constants import TIMEOUT
from .utils import configure_memgpt


def test_configure_memgpt():
    configure_memgpt()


# def test_legacy_cli_sequence():

if __name__ == "__main__":
    test_configure_memgpt()
    # test_legacy_cli_sequence()
