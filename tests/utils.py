import datetime
import os
from datetime import datetime
from importlib import util
from typing import Dict, Iterator, List, Tuple

import requests

from letta.config import LettaConfig
from letta.data_sources.connectors import DataConnector
from letta.schemas.file import FileMetadata
from letta.settings import TestSettings

from .constants import TIMEOUT


class DummyDataConnector(DataConnector):
    """Fake data connector for texting which yields document/passage texts from a provided list"""

    def __init__(self, texts: List[str]):
        self.texts = texts
        self.file_to_text = {}

    def find_files(self, source) -> Iterator[FileMetadata]:
        for text in self.texts:
            file_metadata = FileMetadata(
                source_id=source.id,
                file_name="",
                file_path="",
                file_type="",
                file_size=0,  # Set to 0 as a placeholder
                file_creation_date="1970-01-01",  # Placeholder date
                file_last_modified_date="1970-01-01",  # Placeholder date
                created_at=datetime.utcnow(),
            )
            self.file_to_text[file_metadata.id] = text

            yield file_metadata

    def generate_passages(self, file: FileMetadata, chunk_size: int = 1024) -> Iterator[Tuple[str | Dict]]:
        yield self.file_to_text[file.id], {}


def wipe_config():
    test_settings = TestSettings()
    config_path = os.path.join(test_settings.letta_dir, "config")
    if os.path.exists(config_path):
        # delete
        os.remove(config_path)


def wipe_letta_home():
    """Wipes ~/.letta (moves to a backup), and initializes a new ~/.letta dir"""

    # Get the current timestamp in a readable format (e.g., YYYYMMDD_HHMMSS)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Construct the new backup directory name with the timestamp
    backup_dir = f"~/.letta_test_backup_{timestamp}"

    # Use os.system to execute the 'mv' command
    os.system(f"mv ~/.letta {backup_dir}")

    # Setup the initial directory
    test_settings = TestSettings()
    config_path = os.path.join(test_settings.letta_dir, "config")
    config = LettaConfig(config_path=config_path)
    config.create_config_dir()


def configure_letta_localllm():
    import pexpect

    wipe_config()
    child = pexpect.spawn("letta configure")

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


def configure_letta(enable_openai=False, enable_azure=False):
    if enable_openai:
        raise NotImplementedError
    elif enable_azure:
        raise NotImplementedError
    else:
        configure_letta_localllm()


def qdrant_server_running() -> bool:
    """Check if Qdrant server is running."""

    try:
        response = requests.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") == "qdrant - vector search engine"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


def with_qdrant_storage(storage: list[str]):
    """If Qdrant server is running and `qdrant_client` is installed,
    append `'qdrant'` to the storage list"""

    if util.find_spec("qdrant_client") is not None and qdrant_server_running():
        storage.append("qdrant")

    return storage
