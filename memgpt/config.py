import logging
import logging.config
from memgpt.log import logger, reload_logger, fix_file_path
import inspect
import json
import os
import uuid
from dataclasses import dataclass
import configparser

import memgpt
import memgpt.utils as utils
from memgpt.constants import MEMGPT_DIR, LLM_MAX_TOKENS, DEFAULT_HUMAN, DEFAULT_PERSONA, LOGGER_NAME
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
    archival_storage_type: str = "local"  # local, db
    archival_storage_path: str = None  # TODO: set to memgpt dir
    archival_storage_uri: str = None  # TODO: eventually allow external vector DB

    # database configs: recall
    recall_storage_type: str = "local"  # local, db
    recall_storage_path: str = None  # TODO: set to memgpt dir
    recall_storage_uri: str = None  # TODO: eventually allow external vector DB

    # database configs: agent state
    persistence_manager_type: str = None  # in-memory, db
    persistence_manager_save_file: str = None  # local file
    persistence_manager_uri: str = None  # db URI

    # version (for backcompat)
    memgpt_version: str = None

    # logging (for logger)
    logging_level: str = "CRITICAL"  # default log level
    logging_enable_logfile: bool = True
    logging_backup_count: int = 3
    logging_max_file_bytes: int = 10 * 1024 * 1024  # 10 MB in bytes
    logging_logdir: str = os.path.join(MEMGPT_DIR, "logs")
    logging_logpathname: str = os.path.join(logging_logdir, "memgpt.log")

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

        # insure all configuration directories exist
        cls.create_config_dir()
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
                "anon_clientid": get_field(config, "client", "anon_clientid"),
                "config_path": config_path,
                "memgpt_version": get_field(config, "version", "memgpt_version"),
                "logging_level": get_field(config, "logger_MemGPT", "level"),
                "logging_enable_logfile": True
                if "consoleHandler,logfileHandler" == get_field(config, "logger_MemGPT", "handlers")
                else False,
                "logging_backup_count": get_field(config, "handler_logfileHandler", "backupcount"),
                "logging_max_file_bytes": get_field(config, "handler_logfileHandler", "maxBytes"),
                "logging_logdir": get_field(config, "logging_paths", "logdir"),
                "logging_logpathname": get_field(config, "logging_paths", "logpathname"),
            }
            # ensure logging is config is set correctly support for upgrades
            force_save = False
            if (
                config_dict["logging_level"] is None
                or config_dict["logging_backup_count"] is None
                or config_dict["logging_max_file_bytes"] is None
                or config_dict["logging_logdir"] is None
                or config_dict["logging_logpathname"] is None
            ):
                # load loggind defaults if none
                config_dict["logging_enable_logfile"] = MemGPTConfig.logging_enable_logfile
                config_dict["logging_level"] = MemGPTConfig.logging_level
                config_dict["logging_backup_count"] = MemGPTConfig.logging_backup_count
                config_dict["logging_max_file_bytes"] = MemGPTConfig.logging_max_file_bytes
                config_dict["logging_logdir"] = MemGPTConfig.logging_logdir
                config_dict["logging_logpathname"] = MemGPTConfig.logging_logpathname
                force_save = True
            config_dict = {k: v for k, v in config_dict.items() if v is not None}

            if force_save:
                temp_config = cls(**config_dict)
                temp_config.save()
                logger = logging.getLogger(LOGGER_NAME)
                logger.debug(f"Updated Missing Logging Configuration: {config_path}")

            return cls(**config_dict)

        # create new config
        anon_clientid = MemGPTConfig.generate_uuid()
        config = cls(anon_clientid=anon_clientid, config_path=config_path)
        config.create_config_dir()  # create dirs
        config.save()  # save updated config
        logger = logging.getLogger(LOGGER_NAME)
        logger.debug(f"Created New Configuration: {config_path}")
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

        # set version
        set_field(config, "version", "memgpt_version", memgpt.__version__)

        # client
        if not self.anon_clientid:
            self.anon_clientid = self.generate_uuid()
        set_field(config, "client", "anon_clientid", self.anon_clientid)

        # logging
        set_field(config, "loggers", "keys", "root,MemGPT")
        set_field(config, "handlers", "keys", "consoleHandler,logfileHandler")
        set_field(config, "formatters", "keys", "consoleFormatter,logfileFormatter")
        # logging root possibly used by other modules not using MemGPT logger
        set_field(config, "logger_root", "level", "CRITICAL")
        set_field(config, "logger_root", "handlers", "consoleHandler")
        # logging MemGPT
        set_field(config, "logger_MemGPT", "level", self.logging_level)
        if self.logging_enable_logfile:
            # this will enable logging to file
            set_field(config, "logger_MemGPT", "handlers", "consoleHandler,logfileHandler")
        else:
            # this removes file logging if not enabled
            set_field(config, "logger_MemGPT", "handlers", "consoleHandler")
        set_field(config, "logger_MemGPT", "qualname", "MemGPT")
        set_field(config, "logger_MemGPT", "propagate", "0")  # do not propigate to root
        # console logging handler
        set_field(config, "handler_consoleHandler", "class", "StreamHandler")
        set_field(config, "handler_consoleHandler", "level", self.logging_level)
        set_field(config, "handler_consoleHandler", "formatter", "consoleFormatter")
        set_field(config, "handler_consoleHandler", "args", "(sys.stdout,)")
        # console logging formatter
        set_field(config, "formatter_consoleFormatter", "format", "%(name)s - %(levelname)s - %(message)s")
        if self.logging_enable_logfile:
            set_field(config, "formatter_logfileFormatter", "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            # logfile logging handler Rotating File Handler
            set_field(config, "handler_logfileHandler", "class", "handlers.RotatingFileHandler")
            set_field(config, "handler_logfileHandler", "level", self.logging_level)
            fixed_logpathname = fix_file_path(self.logging_logpathname)
            set_field(
                config,
                "handler_logfileHandler",
                "args",
                f"('{fixed_logpathname}', {self.logging_max_file_bytes}, {self.logging_backup_count})",
            )
            set_field(config, "handler_logfileHandler", "formatter", "logfileFormatter")
        # logging paths
        set_field(config, "logging_paths", "logdir", self.logging_logdir)
        set_field(config, "logging_paths", "logpathname", self.logging_logpathname)

        # always make sure all directories are present
        self.create_config_dir()

        with open(self.config_path, "w") as f:
            config.write(f)
        # reload logging config after write.
        logging.config.fileConfig(self.config_path, disable_existing_loggers=False)
        # reset the logger (global) logger is defined as global
        reload_logger()
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

        folders = ["personas", "humans", "archival", "agents", "functions", "system_prompts", "presets", "settings", "logs"]

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
        model,
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
