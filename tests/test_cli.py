import os
import shutil
import sys

import pexpect
import pytest

from letta.local_llm.constants import (
    ASSISTANT_MESSAGE_CLI_SYMBOL,
    INNER_THOUGHTS_CLI_SYMBOL,
)

original_letta_path = os.path.expanduser("~/.letta")
backup_letta_path = os.path.expanduser("~/.letta_backup")


@pytest.fixture
def swap_letta_config():
    if os.path.exists(backup_letta_path):
        print("\nDelete the backup ~/.letta directory\n")
        shutil.rmtree(backup_letta_path)

    if os.path.exists(original_letta_path):
        print("\nBackup the original ~/.letta directory\n")
        shutil.move(original_letta_path, backup_letta_path)

    try:
        # Run the test
        yield
    finally:
        # Ensure this runs no matter what
        print("\nClean up ~/.letta and restore the original directory\n")
        if os.path.exists(original_letta_path):
            shutil.rmtree(original_letta_path)

        if os.path.exists(backup_letta_path):
            shutil.move(backup_letta_path, original_letta_path)


def test_letta_run_create_new_agent(swap_letta_config):
    child = pexpect.spawn("poetry run letta run", encoding="utf-8")
    # Start the letta run command
    child.logfile = sys.stdout
    child.expect("Creating new agent", timeout=20)
    # Optional: LLM model selection
    try:
        child.expect("Select LLM model:", timeout=20)
        child.sendline("")
    except (pexpect.TIMEOUT, pexpect.EOF):
        print("[WARNING] LLM model selection step was skipped.")

    # Optional: Context window selection
    try:
        child.expect("Select LLM context window limit", timeout=20)
        child.sendline("")
    except (pexpect.TIMEOUT, pexpect.EOF):
        print("[WARNING] Context window selection step was skipped.")

    # Optional: Embedding model selection
    try:
        child.expect("Select embedding model:", timeout=20)
        child.sendline("text-embedding-ada-002")
    except (pexpect.TIMEOUT, pexpect.EOF):
        print("[WARNING] Embedding model selection step was skipped.")

    child.expect("Created new agent", timeout=20)
    child.sendline("")

    # Get initial response
    child.expect("Enter your message:", timeout=60)
    # Capture the output up to this point
    full_output = child.before
    assert full_output is not None, "No output was captured."
    # Count occurrences of inner thoughts
    cloud_emoji_count = full_output.count(INNER_THOUGHTS_CLI_SYMBOL)
    assert cloud_emoji_count == 1, f"It appears that there are multiple instances of inner thought outputted."
    # Count occurrences of assistant messages
    robot = full_output.count(ASSISTANT_MESSAGE_CLI_SYMBOL)
    assert robot == 1, f"It appears that there are multiple instances of assistant messages outputted."
    # Make sure the user name was repeated back at least once
    assert full_output.count("Chad") > 0, f"Chad was not mentioned...please manually inspect the outputs."
