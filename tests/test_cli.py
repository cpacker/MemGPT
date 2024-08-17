import os
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "pexpect"])
import pexpect
from prettytable.colortable import ColorTable

from memgpt.cli.cli_config import ListChoice, add, delete
from memgpt.cli.cli_config import list as list_command

from .constants import TIMEOUT
from .utils import create_config

# def test_configure_memgpt():
#    configure_memgpt()

options = [ListChoice.agents, ListChoice.sources, ListChoice.humans, ListChoice.personas]


def test_cli_list():
    for option in options:
        output = list_command(arg=option)
        # check if is a list
        assert isinstance(output, ColorTable)


def test_cli_config():

    # test add
    for option in ["human", "persona"]:

        # create initial
        add(option=option, name="test", text="test data")

        ## update
        # filename = "test.txt"
        # open(filename, "w").write("test data new")
        # child = pexpect.spawn(f"poetry run memgpt add --{str(option)} {filename} --name test --strip-ui")
        # child.expect("Human test already exists. Overwrite?", timeout=TIMEOUT)
        # child.sendline()
        # child.expect(pexpect.EOF, timeout=TIMEOUT)  # Wait for child to exit
        # child.close()

        for row in list_command(arg=ListChoice.humans if option == "human" else ListChoice.personas):
            if row[0] == "test":
                assert "test data" in row
        # delete
        delete(option=option, name="test")


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
