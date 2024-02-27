from memgpt.log import logger
import inspect
import json
import os
import uuid
from dataclasses import dataclass, field
import configparser

import memgpt
import memgpt.utils as utils

from memgpt.constants import MEMGPT_DIR, DEFAULT_HUMAN, DEFAULT_PERSONA, DEFAULT_PRESET
from memgpt.data_types import AgentState, LLMConfig, EmbeddingConfig


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
    config_path: str = os.getenv("MEMGPT_CONFIG_PATH") or os.path.join(MEMGPT_DIR, "config")
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
    memgpt_version: str = memgpt.__version__

    # user info
    policies_accepted: bool = False

    def __post_init__(self):
        # ensure types
        # self.embedding_chunk_size = int(self.embedding_chunk_size)
        # self.embedding_dim = int(self.embedding_dim)
        # self.context_window = int(self.context_window)
        pass

    @staticmethod
    def generate_uuid() -> str:
        return uuid.UUID(int=uuid.getnode()).hex

    @classmethod
    def load(cls) -> "MemGPTConfig":
        # avoid circular import
        from memgpt.migrate import config_is_compatible, VERSION_CUTOFF

        if not config_is_compatible(allow_empty=True):
            error_message = " ".join(
                [
                    f"\nYour current config file is incompatible with MemGPT versions later than {VERSION_CUTOFF}.",
                    f"\nTo use MemGPT, you must either downgrade your MemGPT version (<= {VERSION_CUTOFF}) or regenerate your config using `memgpt configure`, or `memgpt migrate` if you would like to migrate old agents.",
                ]
            )
            raise ValueError(error_message)

        config = configparser.ConfigParser()

        # allow overriding with env variables
        if os.getenv("MEMGPT_CONFIG_PATH"):
            config_path = os.getenv("MEMGPT_CONFIG_PATH")
        else:
            config_path = MemGPTConfig.config_path

        # insure all configuration directories exist
        cls.create_config_dir()
        if os.path.exists(config_path):
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
                "archival_storage_path": get_field(config, "archival_storage", "path"),
                "archival_storage_uri": get_field(config, "archival_storage", "uri"),
                "recall_storage_type": get_field(config, "recall_storage", "type"),
                "recall_storage_path": get_field(config, "recall_storage", "path"),
                "recall_storage_uri": get_field(config, "recall_storage", "uri"),
                "metadata_storage_type": get_field(config, "metadata_storage", "type"),
                "metadata_storage_path": get_field(config, "metadata_storage", "path"),
                "metadata_storage_uri": get_field(config, "metadata_storage", "uri"),
                # Misc
                "anon_clientid": get_field(config, "client", "anon_clientid"),
                "config_path": config_path,
                "memgpt_version": get_field(config, "version", "memgpt_version"),
            }

            # Don't include null values
            config_dict = {k: v for k, v in config_dict.items() if v is not None}

            return cls(**config_dict)

        # create new config
        anon_clientid = MemGPTConfig.generate_uuid()
        config = cls(anon_clientid=anon_clientid, config_path=config_path)
        config.create_config_dir()  # create dirs

        return config

    def save(self):
        import memgpt

        config = configparser.ConfigParser()

        # CLI defaults
        set_field(config, "defaults", "preset", self.preset)
        set_field(config, "defaults", "persona", self.persona)
        set_field(config, "defaults", "human", self.human)

        # model defaults
        set_field(config, "model", "model", self.default_llm_config.model)
        set_field(config, "model", "model_endpoint", self.default_llm_config.model_endpoint)
        set_field(
            config,
            "model",
            "model_endpoint_type",
            self.default_llm_config.model_endpoint_type,
        )
        set_field(config, "model", "model_wrapper", self.default_llm_config.model_wrapper)
        set_field(
            config,
            "model",
            "context_window",
            str(self.default_llm_config.context_window),
        )

        # embeddings
        set_field(
            config,
            "embedding",
            "embedding_endpoint_type",
            self.default_embedding_config.embedding_endpoint_type,
        )
        set_field(
            config,
            "embedding",
            "embedding_endpoint",
            self.default_embedding_config.embedding_endpoint,
        )
        set_field(
            config,
            "embedding",
            "embedding_model",
            self.default_embedding_config.embedding_model,
        )
        set_field(
            config,
            "embedding",
            "embedding_dim",
            str(self.default_embedding_config.embedding_dim),
        )
        set_field(
            config,
            "embedding",
            "embedding_chunk_size",
            str(self.default_embedding_config.embedding_chunk_size),
        )

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

        # always make sure all directories are present
        self.create_config_dir()

        with open(self.config_path, "w", encoding="utf-8") as f:
            config.write(f)
        logger.debug(f"Saved Config:  {self.config_path}")

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

        folders = [
            "personas",
            "humans",
            "archival",
            "agents",
            "functions",
            "system_prompts",
            "presets",
            "settings",
        ]

        for folder in folders:
            if not os.path.exists(os.path.join(MEMGPT_DIR, folder)):
                os.makedirs(os.path.join(MEMGPT_DIR, folder))


@dataclass
class AgentConfig:
    """

    NOTE: this is a deprecated class, use AgentState instead. This class is only used for backcompatibility.
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
        self.context_window = config.default_llm_config.context_window if context_window is None else context_window
        self.model = config.default_llm_config.model if model is None else model
        self.model_endpoint_type = config.default_llm_config.model_endpoint_type if model_endpoint_type is None else model_endpoint_type
        self.model_endpoint = config.default_llm_config.model_endpoint if model_endpoint is None else model_endpoint
        self.model_wrapper = config.default_llm_config.model_wrapper if model_wrapper is None else model_wrapper
        self.llm_config = LLMConfig(
            model=self.model,
            model_endpoint_type=self.model_endpoint_type,
            model_endpoint=self.model_endpoint,
            model_wrapper=self.model_wrapper,
            context_window=self.context_window,
        )
        self.embedding_endpoint_type = (
            config.default_embedding_config.embedding_endpoint_type if embedding_endpoint_type is None else embedding_endpoint_type
        )
        self.embedding_endpoint = config.default_embedding_config.embedding_endpoint if embedding_endpoint is None else embedding_endpoint
        self.embedding_model = config.default_embedding_config.embedding_model if embedding_model is None else embedding_model
        self.embedding_dim = config.default_embedding_config.embedding_dim if embedding_dim is None else embedding_dim
        self.embedding_chunk_size = (
            config.default_embedding_config.embedding_chunk_size if embedding_chunk_size is None else embedding_chunk_size
        )
        self.embedding_config = EmbeddingConfig(
            embedding_endpoint_type=self.embedding_endpoint_type,
            embedding_endpoint=self.embedding_endpoint,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
            embedding_chunk_size=self.embedding_chunk_size,
        )

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
        with open(self.agent_config_path, "w", encoding="utf-8") as f:
            json.dump(vars(self), f, indent=4)

    def to_agent_state(self):
        return AgentState(
            name=self.name,
            preset=self.preset,
            persona=self.persona,
            human=self.human,
            llm_config=self.llm_config,
            embedding_config=self.embedding_config,
            create_time=self.create_time,
        )

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
        with open(agent_config_path, "r", encoding="utf-8") as f:
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
