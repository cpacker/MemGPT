import datetime
import os

from memgpt.config import MemGPTConfig
from memgpt.cli.cli import quickstart, QuickstartChoice
from memgpt import Admin

from .constants import TIMEOUT


def create_config(endpoint="openai"):
    """Create config file matching quickstart option"""
    if endpoint == "openai":
        quickstart(QuickstartChoice.openai)
    elif endpoint == "memgpt_hosted":
        quickstart(QuickstartChoice.memgpt_hosted)
    else:
        raise ValueError(f"Invalid endpoint {endpoint}")


def wipe_config():
    if MemGPTConfig.exists():
        # delete
        if os.getenv("MEMGPT_CONFIG_PATH"):
            config_path = os.getenv("MEMGPT_CONFIG_PATH")
        else:
            config_path = MemGPTConfig.config_path
        # TODO delete file config_path
        os.remove(config_path)
        assert not MemGPTConfig.exists(), "Config should not exist after deletion"
    else:
        print("No config to wipe", MemGPTConfig.config_path)


def wipe_memgpt_home():
    """Wipes ~/.memgpt (moves to a backup), and initializes a new ~/.memgpt dir"""

    # Get the current timestamp in a readable format (e.g., YYYYMMDD_HHMMSS)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the new backup directory name with the timestamp
    backup_dir = f"~/.memgpt_test_backup_{timestamp}"

    # Use os.system to execute the 'mv' command
    os.system(f"mv ~/.memgpt {backup_dir}")

    # Setup the initial directory
    MemGPTConfig.create_config_dir()


def configure_memgpt_localllm():
    import pexpect

    wipe_config()
    child = pexpect.spawn("memgpt configure")

    child.expect("Select LLM inference provider", timeout=TIMEOUT)
    child.send("\x1b[B")  # Send the down arrow key
    child.send("\x1b[B")  # Send the down arrow key
    child.sendline()

    child.expect("Select LLM backend", timeout=TIMEOUT)
    child.sendline()

    child.expect("Enter default endpoint", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select default model wrapper", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select your model's context window", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select embedding provider", timeout=TIMEOUT)
    child.send("\x1b[B")  # Send the down arrow key
    child.send("\x1b[B")  # Send the down arrow key
    child.send("\x1b[B")  # Send the down arrow key
    child.sendline()

    child.expect("Select default preset", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select default persona", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select default human", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select storage backend for archival data", timeout=TIMEOUT)
    child.sendline()

    child.sendline()

    child.expect(pexpect.EOF, timeout=TIMEOUT)  # Wait for child to exit
    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."


def configure_memgpt(enable_openai=False, enable_azure=False):
    if enable_openai:
        raise NotImplementedError
    elif enable_azure:
        raise NotImplementedError
    else:
        configure_memgpt_localllm()
