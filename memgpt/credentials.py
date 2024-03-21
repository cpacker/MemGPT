from memgpt.log import logger
import os
from dataclasses import dataclass
import configparser
import typer
import questionary
from typing import Optional

import memgpt
import memgpt.utils as utils
from memgpt.utils import printd, get_schema_diff
from memgpt.functions.functions import load_all_function_sets

from memgpt.constants import MEMGPT_DIR, LLM_MAX_TOKENS, DEFAULT_HUMAN, DEFAULT_PERSONA, DEFAULT_PRESET
from memgpt.data_types import AgentState, User, LLMConfig, EmbeddingConfig
from memgpt.config import get_field, set_field


SUPPORTED_AUTH_TYPES = ["bearer_token", "api_key"]
KEY_NAMES = ("openai_key", "azure_key", "openllm_key")


@dataclass
class MemGPTCredentials:
    # key_roller functionality with backward compatability
    def __getattribute__(self, prop: str):
        # exclude processing of all fields except fields with keys
        value = super().__getattribute__(prop)
        if value is None or prop not in KEY_NAMES:
            return value

        # the key field stores comma-separated keys
        keys = [key for key in value.split(",")]
        # calculating the current key number
        number = self._key_num
        number = number + 1 if number + 1 < len(keys) else 0
        self._key_num = number
        # key issuance
        if len(keys):
            return keys[number]
        return None

    def _get_keys(self, key_name):
        if key_name in KEY_NAMES:
            return super().__getattribute__(key_name)

    _key_num: int = 0

    # credentials for MemGPT
    credentials_path: str = os.path.join(MEMGPT_DIR, "credentials")

    # openai config
    openai_auth_type: str = "bearer_token"
    openai_key: Optional[str] = None

    # azure config
    azure_auth_type: str = "api_key"
    azure_key: Optional[str] = None

    # base llm / model
    azure_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    # embeddings
    azure_embedding_version: Optional[str] = None
    azure_embedding_endpoint: Optional[str] = None
    azure_embedding_deployment: Optional[str] = None

    # custom llm API config
    openllm_auth_type: Optional[str] = None
    openllm_key: Optional[str] = None

    @classmethod
    def load(cls) -> "MemGPTCredentials":
        config = configparser.ConfigParser()

        # allow overriding with env variables
        credentials_path = os.getenv("MEMGPT_CREDENTIALS_PATH")
        if not credentials_path:
            credentials_path = MemGPTCredentials.credentials_path

        if os.path.exists(credentials_path):
            # read existing credentials
            config.read(credentials_path)
            config_dict = {
                # openai
                "openai_auth_type": get_field(config, "openai", "auth_type"),
                "openai_key": get_field(config, "openai", "key"),
                # azure
                "azure_auth_type": get_field(config, "azure", "auth_type"),
                "azure_key": get_field(config, "azure", "key"),
                "azure_version": get_field(config, "azure", "version"),
                "azure_endpoint": get_field(config, "azure", "endpoint"),
                "azure_deployment": get_field(config, "azure", "deployment"),
                "azure_embedding_version": get_field(config, "azure", "embedding_version"),
                "azure_embedding_endpoint": get_field(config, "azure", "embedding_endpoint"),
                "azure_embedding_deployment": get_field(config, "azure", "embedding_deployment"),
                # open llm
                "openllm_auth_type": get_field(config, "openllm", "auth_type"),
                "openllm_key": get_field(config, "openllm", "key"),
                # path
                "credentials_path": credentials_path,
            }
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            return cls(**config_dict)

        # create new config
        config = cls(credentials_path=credentials_path)
        config.save()  # save updated config
        return config

    def save(self):
        import memgpt

        config = configparser.ConfigParser()
        # openai config
        set_field(config, "openai", "auth_type", self.openai_auth_type)
        set_field(config, "openai", "key", self._get_keys("openai_key"))

        # azure config
        set_field(config, "azure", "auth_type", self.azure_auth_type)
        set_field(config, "azure", "key", self._get_keys("azure_key"))
        set_field(config, "azure", "version", self.azure_version)
        set_field(config, "azure", "endpoint", self.azure_endpoint)
        set_field(config, "azure", "deployment", self.azure_deployment)
        set_field(config, "azure", "embedding_version", self.azure_embedding_version)
        set_field(config, "azure", "embedding_endpoint", self.azure_embedding_endpoint)
        set_field(config, "azure", "embedding_deployment", self.azure_embedding_deployment)

        # openllm config
        set_field(config, "openllm", "auth_type", self.openllm_auth_type)
        set_field(config, "openllm", "key", self._get_keys("openllm_key"))

        if not os.path.exists(MEMGPT_DIR):
            os.makedirs(MEMGPT_DIR, exist_ok=True)
        with open(self.credentials_path, "w", encoding="utf-8") as f:
            config.write(f)

    @staticmethod
    def exists():
        # allow overriding with env variables
        credentials_path = os.getenv("MEMGPT_CREDENTIALS_PATH")
        if not credentials_path:
            credentials_path = MemGPTCredentials.credentials_path

        assert not os.path.isdir(credentials_path), f"Credentials path {credentials_path} cannot be set to a directory."
        return os.path.exists(credentials_path)
