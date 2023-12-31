import inspect
import json
import os
import uuid
from dataclasses import dataclass
import configparser

import memgpt
import memgpt.utils as utils
from memgpt.utils import printd, get_schema_diff
from memgpt.functions.functions import load_all_function_sets
from memgpt.constants import MEMGPT_DIR, LLM_MAX_TOKENS, DEFAULT_HUMAN, DEFAULT_PERSONA
from memgpt.presets.presets import DEFAULT_PRESET


# helper functions for writing to configs
def get_field(config, section, field):
    if section not in config:
        return None
    if config.has_option(section, field):
        return config.get(section, field)
    else:
        return None


def set_field(config, section, field, value):
    if value is None:  # cannot write None
        return
    if section not in config:  # create section
        config.add_section(section)
    config.set(section, field, value)


@dataclass
class MemGPTConfig:
    config_path: str = os.path.join(MEMGPT_DIR, "config")
    anon_clientid: str = None

    # preset
    preset: str = DEFAULT_PRESET

    # model parameters
    model: str = None
    model_endpoint_type: str = None
    model_endpoint: str = None  # localhost:8000
    model_wrapper: str = None
    context_window: int = LLM_MAX_TOKENS[model] if model in LLM_MAX_TOKENS else LLM_MAX_TOKENS["DEFAULT"]

    # model parameters: openai
    openai_key: str = None

    # model parameters: azure
    azure_key: str = None
    azure_endpoint: str = None
    azure_version: str = None
    azure_deployment: str = None
    azure_embedding_deployment: str = None

    # persona parameters
    persona: str = DEFAULT_PERSONA
    human: str = DEFAULT_HUMAN
    agent: str = None

    # embedding parameters
    embedding_endpoint_type: str = "openai"  # openai, azure, local
    embedding_endpoint: str = None
    embedding_model: str = None
    embedding_dim: int = 1536
    embedding_chunk_size: int = 300  # number of tokens

    # database configs: archival
    archival_storage_type: str = "chroma"  # local, db
    archival_storage_path: str = os.path.join(MEMGPT_DIR, "chroma")
    archival_storage_uri: str = None  # TODO: eventually allow external vector DB

    # database configs: recall
    recall_storage_type: str = "sqlite"  # local, db
    recall_storage_path: str = MEMGPT_DIR
    recall_storage_uri: str = None  # TODO: eventually allow external vector DB

    # database configs: metadata storage (sources, agents, data sources)
    metadata_storage_type: str = "sqlite"
    metadata_storage_path: str = MEMGPT_DIR
    metadata_storage_uri: str = None

    # database configs: agent state
    persistence_manager_type: str = None  # in-memory, db
    persistence_manager_save_file: str = None  # local file
    persistence_manager_uri: str = None  # db URI

    # version (for backcompat)
    memgpt_version: str = None

    # user info
    policies_accepted: bool = False

    def __post_init__(self):
        # ensure types
        self.embedding_chunk_size = int(self.embedding_chunk_size)
        self.embedding_dim = int(self.embedding_dim)
        self.context_window = int(self.context_window)

    @staticmethod
    def generate_uuid() -> str:
        return uuid.UUID(int=uuid.getnode()).hex

    @classmethod
    def load(cls) -> "MemGPTConfig":
        config = configparser.ConfigParser()

        # allow overriding with env variables
        if os.getenv("MEMGPT_CONFIG_PATH"):
            config_path = os.getenv("MEMGPT_CONFIG_PATH")
        else:
            config_path = MemGPTConfig.config_path

        if os.path.exists(config_path):
            # read existing config
            config.read(config_path)
            config_dict = {
                "model": get_field(config, "model", "model"),
                "model_endpoint": get_field(config, "model", "model_endpoint"),
                "model_endpoint_type": get_field(config, "model", "model_endpoint_type"),
                "model_wrapper": get_field(config, "model", "model_wrapper"),
                "context_window": get_field(config, "model", "context_window"),
                "preset": get_field(config, "defaults", "preset"),
                "persona": get_field(config, "defaults", "persona"),
                "human": get_field(config, "defaults", "human"),
                "agent": get_field(config, "defaults", "agent"),
                "openai_key": get_field(config, "openai", "key"),
                "azure_key": get_field(config, "azure", "key"),
                "azure_endpoint": get_field(config, "azure", "endpoint"),
                "azure_version": get_field(config, "azure", "version"),
                "azure_deployment": get_field(config, "azure", "deployment"),
                "azure_embedding_deployment": get_field(config, "azure", "embedding_deployment"),
                "embedding_endpoint": get_field(config, "embedding", "embedding_endpoint"),
                "embedding_model": get_field(config, "embedding", "embedding_model"),
                "embedding_endpoint_type": get_field(config, "embedding", "embedding_endpoint_type"),
                "embedding_dim": get_field(config, "embedding", "embedding_dim"),
                "embedding_chunk_size": get_field(config, "embedding", "chunk_size"),
                "archival_storage_type": get_field(config, "archival_storage", "type"),
                "archival_storage_path": get_field(config, "archival_storage", "path"),
                "archival_storage_uri": get_field(config, "archival_storage", "uri"),
                "recall_storage_type": get_field(config, "recall_storage", "type"),
                "recall_storage_path": get_field(config, "recall_storage", "path"),
                "recall_storage_uri": get_field(config, "recall_storage", "uri"),
                "metadata_storage_type": get_field(config, "metadata_storage", "type"),
                "metadata_storage_path": get_field(config, "metadata_storage", "path"),
                "metadata_storage_uri": get_field(config, "metadata_storage", "uri"),
                "anon_clientid": get_field(config, "client", "anon_clientid"),
                "config_path": config_path,
                "memgpt_version": get_field(config, "version", "memgpt_version"),
            }
            config_dict = {k: v for k, v in config_dict.items() if v is not None}
            return cls(**config_dict)

        # create new config
        anon_clientid = MemGPTConfig.generate_uuid()
        config = cls(anon_clientid=anon_clientid, config_path=config_path)
        config.save()  # save updated config
        return config

    def save(self):
        import memgpt

        config = configparser.ConfigParser()

        # CLI defaults
        set_field(config, "defaults", "preset", self.preset)
        set_field(config, "defaults", "persona", self.persona)
        set_field(config, "defaults", "human", self.human)
        set_field(config, "defaults", "agent", self.agent)

        # model defaults
        set_field(config, "model", "model", self.model)
        set_field(config, "model", "model_endpoint", self.model_endpoint)
        set_field(config, "model", "model_endpoint_type", self.model_endpoint_type)
        set_field(config, "model", "model_wrapper", self.model_wrapper)
        set_field(config, "model", "context_window", str(self.context_window))

        # security credentials: openai
        set_field(config, "openai", "key", self.openai_key)

        # security credentials: azure
        set_field(config, "azure", "key", self.azure_key)
        set_field(config, "azure", "endpoint", self.azure_endpoint)
        set_field(config, "azure", "version", self.azure_version)
        set_field(config, "azure", "deployment", self.azure_deployment)
        set_field(config, "azure", "embedding_deployment", self.azure_embedding_deployment)

        # embeddings
        set_field(config, "embedding", "embedding_endpoint_type", self.embedding_endpoint_type)
        set_field(config, "embedding", "embedding_endpoint", self.embedding_endpoint)
        set_field(config, "embedding", "embedding_model", self.embedding_model)
        set_field(config, "embedding", "embedding_dim", str(self.embedding_dim))
        set_field(config, "embedding", "embedding_chunk_size", str(self.embedding_chunk_size))

        # archival storage
        set_field(config, "archival_storage", "type", self.archival_storage_type)
        set_field(config, "archival_storage", "path", self.archival_storage_path)
        set_field(config, "archival_storage", "uri", self.archival_storage_uri)

        # recall storage
        set_field(config, "recall_storage", "type", self.recall_storage_type)
        set_field(config, "recall_storage", "path", self.recall_storage_path)
        set_field(config, "recall_storage", "uri", self.recall_storage_uri)

        # metadata storage
        set_field(config, "metadata_storage", "type", self.metadata_storage_type)
        set_field(config, "metadata_storage", "path", self.metadata_storage_path)
        set_field(config, "metadata_storage", "uri", self.metadata_storage_uri)

        # set version
        set_field(config, "version", "memgpt_version", memgpt.__version__)

        # client
        if not self.anon_clientid:
            self.anon_clientid = self.generate_uuid()
        set_field(config, "client", "anon_clientid", self.anon_clientid)

        if not os.path.exists(MEMGPT_DIR):
            os.makedirs(MEMGPT_DIR, exist_ok=True)
        with open(self.config_path, "w") as f:
            config.write(f)

    @staticmethod
    def exists():
        # allow overriding with env variables
        if os.getenv("MEMGPT_CONFIG_PATH"):
            config_path = os.getenv("MEMGPT_CONFIG_PATH")
        else:
            config_path = MemGPTConfig.config_path

        assert not os.path.isdir(config_path), f"Config path {config_path} cannot be set to a directory."
        return os.path.exists(config_path)

    @staticmethod
    def create_config_dir():
        if not os.path.exists(MEMGPT_DIR):
            os.makedirs(MEMGPT_DIR, exist_ok=True)

        folders = ["personas", "humans", "archival", "agents", "functions", "system_prompts", "presets", "settings"]
        for folder in folders:
            if not os.path.exists(os.path.join(MEMGPT_DIR, folder)):
                os.makedirs(os.path.join(MEMGPT_DIR, folder))


@dataclass
class AgentConfig:
    """
    Configuration for a specific instance of an agent
    """

    def __init__(
        self,
        persona,
        human,
        # model info
        model=None,
        model_endpoint_type=None,
        model_endpoint=None,
        model_wrapper=None,
        context_window=None,
        # embedding info
        embedding_endpoint_type=None,
        embedding_endpoint=None,
        embedding_model=None,
        embedding_dim=None,
        embedding_chunk_size=None,
        # other
        preset=None,
        data_sources=None,
        # agent info
        agent_config_path=None,
        name=None,
        create_time=None,
        memgpt_version=None,
        # functions
        functions=None,  # schema definitions ONLY (linked at runtime)
    ):
        if name is None:
            self.name = f"agent_{self.generate_agent_id()}"
        else:
            self.name = name

        config = MemGPTConfig.load()  # get default values
        self.persona = config.persona if persona is None else persona
        self.human = config.human if human is None else human
        self.preset = config.preset if preset is None else preset
        self.context_window = config.context_window if context_window is None else context_window
        self.model = config.model if model is None else model
        self.model_endpoint_type = config.model_endpoint_type if model_endpoint_type is None else model_endpoint_type
        self.model_endpoint = config.model_endpoint if model_endpoint is None else model_endpoint
        self.model_wrapper = config.model_wrapper if model_wrapper is None else model_wrapper
        self.embedding_endpoint_type = config.embedding_endpoint_type if embedding_endpoint_type is None else embedding_endpoint_type
        self.embedding_endpoint = config.embedding_endpoint if embedding_endpoint is None else embedding_endpoint
        self.embedding_model = config.embedding_model if embedding_model is None else embedding_model
        self.embedding_dim = config.embedding_dim if embedding_dim is None else embedding_dim
        self.embedding_chunk_size = config.embedding_chunk_size if embedding_chunk_size is None else embedding_chunk_size

        # agent metadata
        self.data_sources = data_sources if data_sources is not None else []
        self.create_time = create_time if create_time is not None else utils.get_local_time()
        if memgpt_version is None:
            import memgpt

            self.memgpt_version = memgpt.__version__
        else:
            self.memgpt_version = memgpt_version

        # functions
        self.functions = functions

        # save agent config
        self.agent_config_path = (
            os.path.join(MEMGPT_DIR, "agents", self.name, "config.json") if agent_config_path is None else agent_config_path
        )

    def generate_agent_id(self, length=6):
        ## random character based
        # characters = string.ascii_lowercase + string.digits
        # return ''.join(random.choices(characters, k=length))

        # count based
        agent_count = len(utils.list_agent_config_files())
        return str(agent_count + 1)

    def attach_data_source(self, data_source: str):
        # TODO: add warning that only once source can be attached
        # i.e. previous source will be overriden
        self.data_sources.append(data_source)
        self.save()

    def save_dir(self):
        return os.path.join(MEMGPT_DIR, "agents", self.name)

    def save_state_dir(self):
        # directory to save agent state
        return os.path.join(MEMGPT_DIR, "agents", self.name, "agent_state")

    def save_persistence_manager_dir(self):
        # directory to save persistent manager state
        return os.path.join(MEMGPT_DIR, "agents", self.name, "persistence_manager")

    def save_agent_index_dir(self):
        # save llama index inside of persistent manager directory
        return os.path.join(self.save_persistence_manager_dir(), "index")

    def save(self):
        # save state of persistence manager
        os.makedirs(os.path.join(MEMGPT_DIR, "agents", self.name), exist_ok=True)
        # save version
        self.memgpt_version = memgpt.__version__
        with open(self.agent_config_path, "w") as f:
            json.dump(vars(self), f, indent=4)

    @staticmethod
    def exists(name: str):
        """Check if agent config exists"""
        agent_config_path = os.path.join(MEMGPT_DIR, "agents", name)
        return os.path.exists(agent_config_path)

    @classmethod
    def load(cls, name: str):
        """Load agent config from JSON file"""
        agent_config_path = os.path.join(MEMGPT_DIR, "agents", name, "config.json")
        assert os.path.exists(agent_config_path), f"Agent config file does not exist at {agent_config_path}"
        with open(agent_config_path, "r") as f:
            agent_config = json.load(f)
        # allow compatibility accross versions
        try:
            class_args = inspect.getargspec(cls.__init__).args
        except AttributeError:
            # https://github.com/pytorch/pytorch/issues/15344
            class_args = inspect.getfullargspec(cls.__init__).args
        agent_fields = list(agent_config.keys())
        for key in agent_fields:
            if key not in class_args:
                utils.printd(f"Removing missing argument {key} from agent config")
                del agent_config[key]
        return cls(**agent_config)
