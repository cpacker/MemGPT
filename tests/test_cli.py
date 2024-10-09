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
cached_letta_path = os.path.join("tests", "data", "cached_letta")
backup_letta_path = os.path.expanduser("~/.letta_backup")


def create_credentials_file(parent_dir):
    # Get the API key from the environment
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if the environment variable is missing
    if not openai_api_key:
        raise EnvironmentError("Missing environment variable: OPENAI_API_KEY")

    # Construct the content of the file
    content = f"""
[openai]
auth_type = bearer_token
key = {openai_api_key}
"""

    # Write the content to a file named 'credentials'
    with open(os.path.join(parent_dir, "credentials"), "w") as file:
        file.write(content.strip())

    print("Credentials file created successfully.")


@pytest.fixture
def swap_letta_config():
    print("\nBackup the original ~/.letta directory\n")
    if os.path.exists(original_letta_path):
        shutil.move(original_letta_path, backup_letta_path)

    print("\nCopy the cached .letta directory to ~/.letta\n")
    shutil.copytree(cached_letta_path, original_letta_path)

    print("\nConstruct the credentials file from env.")
    expanded_parent_dir = os.path.expanduser("~/.letta")
    create_credentials_file(expanded_parent_dir)

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


def test_letta_run_existing_agent(swap_letta_config):
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

    # Count occurrences of inner thoughts
    cloud_emoji_count = full_output.count(INNER_THOUGHTS_CLI_SYMBOL)
    assert cloud_emoji_count == 1, f"It appears that there are multiple instances of inner thought outputted."

    # Count occurrences assistant messages
    robot = full_output.count(ASSISTANT_MESSAGE_CLI_SYMBOL)
    assert robot == 1, f"It appears that there are multiple instances of assistant messages outputted."

    # Make sure the user name was repeated back at least once
    assert full_output.count("Chad") > 0, f"Chad was not mentioned...please manually inspect the outputs."
