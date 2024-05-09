import os
from typing import Set
import uuid
from dataclasses import dataclass, field
import configparser

import memgpt

import memgpt.functions.function_sets.base

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA, DEFAULT_PRESET
from memgpt.data_types import LLMConfig, EmbeddingConfig


# helper functions for writing to configs
def get_field(config, section, field):
    if section not in config:
        return None
    if config.has_option(section, field):
        return config.get(section, field)
    else:
        return None


@dataclass
class MemGPTConfig:
    config_path: str = os.environ["MEMGPT_CONFIG_PATH"]
    anon_clientid: str = str(uuid.UUID(int=0))

    # preset
    preset: str = DEFAULT_PRESET

    # persona parameters
    persona: str = DEFAULT_PERSONA
    human: str = DEFAULT_HUMAN

    # model parameters
    default_llm_config: LLMConfig = field(default_factory=LLMConfig)

    # embedding parameters
    default_embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    # database configs: recall
    recall_storage_type: str = ""  # local, db
    recall_storage_uri: str = ""

    # database configs: metadata storage (sources, agents, data sources)
    metadata_storage_type: str = ""
    metadata_storage_uri: str = ""

    archival_storage_type: str = ""
    archival_storage_uri: str = ""

    # full names of modules
    functions_modules: Set[str] = field(default_factory=set)

    @classmethod
    def load(cls) -> "MemGPTConfig":

        config = configparser.ConfigParser()

        # set by env var
        config_path = cls.config_path

        # insure all configuration directories exist
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        # read existing config
        config.read(config_path)

        # Handle extraction of nested LLMConfig and EmbeddingConfig
        llm_config_dict = {
            # Extract relevant LLM configuration from the config file
            "model": get_field(config, "model", "model"),
            "model_endpoint": get_field(config, "model", "model_endpoint"),
            "model_endpoint_type": get_field(config, "model", "model_endpoint_type"),
            "model_wrapper": get_field(config, "model", "model_wrapper"),
            "context_window": get_field(config, "model", "context_window"),
        }
        embedding_config_dict = {
            # Extract relevant Embedding configuration from the config file
            "embedding_endpoint": get_field(config, "embedding", "embedding_endpoint"),
            "embedding_model": get_field(config, "embedding", "embedding_model"),
            "embedding_endpoint_type": get_field(config, "embedding", "embedding_endpoint_type"),
            "embedding_dim": get_field(config, "embedding", "embedding_dim"),
            "embedding_chunk_size": get_field(config, "embedding", "embedding_chunk_size"),
        }
        # Remove null values
        llm_config_dict = {k: v for k, v in llm_config_dict.items() if v is not None}
        embedding_config_dict = {k: v for k, v in embedding_config_dict.items() if v is not None}
        # Correct the types that aren't strings
        if llm_config_dict["context_window"] is not None:
            llm_config_dict["context_window"] = int(llm_config_dict["context_window"])
        if embedding_config_dict["embedding_dim"] is not None:
            embedding_config_dict["embedding_dim"] = int(embedding_config_dict["embedding_dim"])
        if embedding_config_dict["embedding_chunk_size"] is not None:
            embedding_config_dict["embedding_chunk_size"] = int(embedding_config_dict["embedding_chunk_size"])
        # Construct the inner properties
        llm_config = LLMConfig(**llm_config_dict)
        embedding_config = EmbeddingConfig(**embedding_config_dict)

        if config.has_option("functions", "modules"):
            function_modules = {m.strip() for m in config.get("functions", "modules").split(",")}
        else:
            function_modules = set()
        function_modules.add(memgpt.functions.function_sets.base.__name__)

        # Everything else
        config_dict = {
            # Two prepared configs
            "default_llm_config": llm_config,
            "default_embedding_config": embedding_config,
            # Agent related
            "preset": get_field(config, "defaults", "preset"),
            "persona": get_field(config, "defaults", "persona"),
            "human": get_field(config, "defaults", "human"),
            "agent": get_field(config, "defaults", "agent"),
            # Storage related
            "archival_storage_type": get_field(config, "archival_storage", "type"),
            "archival_storage_uri": get_field(config, "archival_storage", "uri"),
            "recall_storage_type": get_field(config, "recall_storage", "type"),
            "recall_storage_uri": get_field(config, "recall_storage", "uri"),
            "metadata_storage_type": get_field(config, "metadata_storage", "type"),
            "metadata_storage_uri": get_field(config, "metadata_storage", "uri"),
            # Misc
            "anon_clientid": get_field(config, "client", "anon_clientid"),
            "functions_modules": function_modules,
        }

        if os.environ.get("POSTGRES_URL"):
            config_dict["recall_storage_uri"] = os.environ.get("POSTGRES_URL")
            config_dict["metadata_storage_uri"] = os.environ.get("POSTGRES_URL")
            config_dict["archival_storage_uri"] = os.environ.get("POSTGRES_URL")

        # Don't include null values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return cls(**config_dict)
