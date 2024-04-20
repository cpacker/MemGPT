import os

from memgpt.config import MemGPTConfig
from memgpt.constants import MEMGPT_DIR


class TestMGPTConfig(MemGPTConfig):
    config_path: str = os.getenv("TEST_MEMGPT_CONFIG_PATH") or os.getenv("MEMGPT_CONFIG_PATH") or os.path.join(MEMGPT_DIR, "config")
