import os
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pexpect"])
import pexpect

from .constants import TIMEOUT
from .utils import create_config

# def test_configure_memgpt():
#    configure_memgpt()


def test_save_load():
    # configure_memgpt()  # rely on configure running first^
    if os.getenv("OPENAI_API_KEY"):
        create_config("openai")
    else:
        create_config("memgpt_hosted")

    child = pexpect.spawn("poetry run memgpt run --agent test_save_load --first --strip-ui")

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

    child = pexpect.spawn("poetry run memgpt run --agent test_save_load --first --strip-ui")
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
