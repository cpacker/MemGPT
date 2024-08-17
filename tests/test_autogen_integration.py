import os
import subprocess

import pytest


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Missing OpenAI API key")
def test_agent_groupchat():

    # Define the path to the script you want to test
    script_path = "memgpt/autogen/examples/agent_groupchat.py"

    # Dynamically get the project's root directory (assuming this script is run from the root)
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # print(project_root)
    # project_root = os.path.join(project_root, "MemGPT")
    # print(project_root)
    # sys.exit(1)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = os.path.join(project_root, "memgpt")
    print(f"Adding the following to PATH: {project_root}")

    # Prepare the environment, adding the project root to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # Run the script using subprocess.run
    # Capture the output (stdout) and the exit code
    # result = subprocess.run(["python", script_path], capture_output=True, text=True)
    result = subprocess.run(["poetry", "run", "python", script_path], capture_output=True, text=True)

    # Check the exit code (0 indicates success)
    assert result.returncode == 0, f"Script exited with code {result.returncode}: {result.stderr}"

    # Optionally, check the output for expected content
    # For example, if you expect a specific line in the output, uncomment and adapt the following line:
    # assert "expected output" in result.stdout, "Expected output not found in script's output"
