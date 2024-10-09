import os
import shutil
import sys

import pexpect
import pytest

original_letta_path = os.path.expanduser("~/.letta")
cached_letta_path = os.path.join("tests", "data", "cached_letta")
backup_letta_path = os.path.expanduser("~/.letta_backup")


@pytest.fixture
def swap_letta_config():
    print("\nBackup the original ~/.letta directory\n")
    if os.path.exists(original_letta_path):
        shutil.move(original_letta_path, backup_letta_path)

    print("\nCopy the cached .letta directory to ~/.letta\n")
    shutil.copytree(cached_letta_path, original_letta_path)

    # Run the test
    yield

    print("\nClean up ~/.letta and restore the original directory\n")
    if os.path.exists(original_letta_path):
        shutil.rmtree(original_letta_path)

    if os.path.exists(backup_letta_path):
        shutil.move(backup_letta_path, original_letta_path)


def test_letta_run_basic(swap_letta_config):
    # Start the letta run command
    child = pexpect.spawn("letta run", encoding="utf-8")
    child.logfile = sys.stdout

    # Select agent
    child.expect("Would you like to select an existing agent?", timeout=3)
    child.sendline("Y")
    child.expect("Select agent:", timeout=3)
    child.sendline("")

    # Agent selected
    child.expect("Using existing agent", timeout=3)
    child.sendline("")

    # Get initial response
    child.expect("Enter your message:", timeout=60)

    # Capture the output up to this point
    full_output = child.before

    # NOTE: These are brittle and may fail if we change the styling of the CLI
    # Count occurrences of the cloud emoji (💭)
    cloud_emoji_count = full_output.count("💭")
    # Assert that the cloud emoji appears only once
    assert cloud_emoji_count == 1, f"It appears that there are multiple instances of inner thought outputted."

    # Count occurrences of the robot emoji (🤖)
    robot = full_output.count("🤖")
    # Assert that the robot emoji appears only once
    assert robot == 1, f"It appears that there are multiple instances of send_message outputted."
