import subprocess
import os
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pexpect"])
import pexpect

from .constants import TIMEOUT
from .utils import configure_memgpt
from memgpt import MemGPT


# def test_configure_memgpt():
#    configure_memgpt()


def test_save_load():
    # configure_memgpt()  # rely on configure running first^
    client = MemGPT(quickstart="openai")

    child = pexpect.spawn("memgpt run --agent test_save_load --first --strip-ui")

    child.expect("Enter your message:", timeout=TIMEOUT)
    child.sendline()

    child.expect("Empty input received. Try again!", timeout=TIMEOUT)
    child.sendline("/save")

    child.expect("Enter your message:", timeout=TIMEOUT)
    child.sendline("/exit")

    child.expect(pexpect.EOF, timeout=TIMEOUT)  # Wait for child to exit
    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."

    child = pexpect.spawn("memgpt run --agent test_save_load --first --strip-ui")
    child.expect("Using existing agent test_save_load", timeout=TIMEOUT)
    child.expect("Enter your message:", timeout=TIMEOUT)
    child.sendline("/exit")
    child.expect(pexpect.EOF, timeout=TIMEOUT)  # Wait for child to exit
    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."


if __name__ == "__main__":
    # test_configure_memgpt()
    test_save_load()
