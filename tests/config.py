import os

from letta.config import LettaConfig
from letta.constants import LETTA_DIR


class TestMGPTConfig(LettaConfig):
    config_path: str = os.getenv("TEST_MEMGPT_CONFIG_PATH") or os.getenv("MEMGPT_CONFIG_PATH") or os.path.join(LETTA_DIR, "config")
