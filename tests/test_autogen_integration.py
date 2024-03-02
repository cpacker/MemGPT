import subprocess
import pytest


def test_agent_groupchat():
    # Define the path to the script you want to test
    script_path = "memgpt/autogen/examples/agent_groupchat.py"

    # Run the script using subprocess.run
    # Capture the output (stdout) and the exit code
    # result = subprocess.run(["python", script_path], capture_output=True, text=True)
    result = subprocess.run(["poetry", "run", "python", script_path], capture_output=True, text=True)

    # Check the exit code (0 indicates success)
    assert result.returncode == 0, f"Script exited with code {result.returncode}: {result.stderr}"

    # Optionally, check the output for expected content
    # For example, if you expect a specific line in the output, uncomment and adapt the following line:
    # assert "expected output" in result.stdout, "Expected output not found in script's output"
