import glob
import inspect
import random
import string
import json
import os
import uuid
import textwrap
from dataclasses import dataclass
import configparser


import questionary

from colorama import Fore, Style

from typing import List, Type

import memgpt
import memgpt.utils as utils
from memgpt.interface import CLIInterface as interface
from memgpt.personas.personas import get_persona_text
from memgpt.humans.humans import get_human_text
from memgpt.constants import MEMGPT_DIR, LLM_MAX_TOKENS
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.presets.presets import DEFAULT_PRESET, preset_options


model_choices = [
    questionary.Choice("gpt-4"),
    questionary.Choice(
        "gpt-4-turbo (developer preview)",
        value="gpt-4-1106-preview",
    ),
    questionary.Choice(
        "gpt-3.5-turbo (experimental! function-calling performance is not quite at the level of gpt-4 yet)",
        value="gpt-3.5-turbo-16k",
    ),
]


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
    persona: str = personas.DEFAULT
    human: str = humans.DEFAULT
    agent: str = None

    # embedding parameters
    embedding_endpoint_type: str = "openai"  # openai, azure, local
    embedding_endpoint: str = None
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
                "embedding_endpoint_type": get_field(config, "embedding", "embedding_endpoint_type"),
                "embedding_dim": get_field(config, "embedding", "embedding_dim"),
                "embedding_chunk_size": get_field(config, "embedding", "chunk_size"),
                "archival_storage_type": get_field(config, "archival_storage", "type"),
                "archival_storage_path": get_field(config, "archival_storage", "path"),
                "archival_storage_uri": get_field(config, "archival_storage", "uri"),
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

        folders = ["personas", "humans", "archival", "agents", "functions", "system_prompts", "presets"]
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
        # assert not os.path.exists(self.agent_config_path), f"Agent config file already exists at {self.agent_config_path}"
        self.save()

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


class Config:
    personas_dir = os.path.join("memgpt", "personas", "examples")
    custom_personas_dir = os.path.join(MEMGPT_DIR, "personas")
    humans_dir = os.path.join("memgpt", "humans", "examples")
    custom_humans_dir = os.path.join(MEMGPT_DIR, "humans")
    configs_dir = os.path.join(MEMGPT_DIR, "configs")

    def __init__(self):
        os.makedirs(Config.custom_personas_dir, exist_ok=True)
        os.makedirs(Config.custom_humans_dir, exist_ok=True)
        self.load_type = None
        self.archival_storage_files = None
        self.compute_embeddings = False
        self.agent_save_file = None
        self.persistence_manager_save_file = None
        self.host = os.getenv("OPENAI_API_BASE")
        self.index = None
        self.config_file = None
        self.preload_archival = False

    @classmethod
    def legacy_flags_init(
        cls: Type["Config"],
        model: str,
        memgpt_persona: str,
        human_persona: str,
        load_type: str = None,
        archival_storage_files: str = None,
        archival_storage_index: str = None,
        compute_embeddings: bool = False,
    ):
        self = cls()
        self.model = model
        self.memgpt_persona = memgpt_persona
        self.human_persona = human_persona
        self.load_type = load_type
        self.archival_storage_files = archival_storage_files
        self.archival_storage_index = archival_storage_index
        self.compute_embeddings = compute_embeddings
        recompute_embeddings = self.compute_embeddings
        if self.archival_storage_index:
            recompute_embeddings = False  # TODO Legacy support -- can't recompute embeddings on a path that's not specified.
        if self.archival_storage_files:
            self.configure_archival_storage(recompute_embeddings)
        return self

    @classmethod
    def config_init(cls: Type["Config"], config_file: str = None):
        self = cls()
        self.config_file = config_file
        if self.config_file is None:
            cfg = Config.get_most_recent_config()
            use_cfg = False
            if cfg:
                print(f"{Style.BRIGHT}{Fore.MAGENTA}⚙️ Found saved config file.{Style.RESET_ALL}")
                use_cfg = questionary.confirm(f"Use most recent config file '{cfg}'?").ask()
            if use_cfg:
                self.config_file = cfg

        if self.config_file:
            self.load_config(self.config_file)
            recompute_embeddings = False
            if self.compute_embeddings:
                if self.archival_storage_index:
                    recompute_embeddings = questionary.confirm(
                        f"Would you like to recompute embeddings? Do this if your files have changed.\n    Files: {self.archival_storage_files}",
                        default=False,
                    ).ask()
                else:
                    recompute_embeddings = True
            if self.load_type:
                self.configure_archival_storage(recompute_embeddings)
                self.write_config()
            return self

        # print("No settings file found, configuring MemGPT...")
        print(f"{Style.BRIGHT}{Fore.MAGENTA}⚙️ No settings file found, configuring MemGPT...{Style.RESET_ALL}")

        self.model = questionary.select(
            "Which model would you like to use?",
            model_choices,
            default=model_choices[0],
        ).ask()

        self.memgpt_persona = questionary.select(
            "Which persona would you like MemGPT to use?",
            Config.get_memgpt_personas(),
        ).ask()
        print(self.memgpt_persona)

        self.human_persona = questionary.select(
            "Which user would you like to use?",
            Config.get_user_personas(),
        ).ask()

        self.archival_storage_index = None
        self.preload_archival = questionary.confirm(
            "Would you like to preload anything into MemGPT's archival memory?", default=False
        ).ask()
        if self.preload_archival:
            self.load_type = questionary.select(
                "What would you like to load?",
                choices=[
                    questionary.Choice("A folder or file", value="folder"),
                    questionary.Choice("A SQL database", value="sql"),
                    questionary.Choice("A glob pattern", value="glob"),
                ],
            ).ask()
            if self.load_type == "folder" or self.load_type == "sql":
                archival_storage_path = questionary.path("Please enter the folder or file (tab for autocomplete):").ask()
                if os.path.isdir(archival_storage_path):
                    self.archival_storage_files = os.path.join(archival_storage_path, "*")
                else:
                    self.archival_storage_files = archival_storage_path
            else:
                self.archival_storage_files = questionary.path("Please enter the glob pattern (tab for autocomplete):").ask()
            self.compute_embeddings = questionary.confirm(
                "Would you like to compute embeddings over these files to enable embeddings search?"
            ).ask()
            self.configure_archival_storage(self.compute_embeddings)

        self.write_config()
        return self

    def configure_archival_storage(self, recompute_embeddings):
        if recompute_embeddings:
            if self.host:
                interface.warning_message(
                    "⛔️ Embeddings on a non-OpenAI endpoint are not yet supported, falling back to substring matching search."
                )
            else:
                self.archival_storage_index = utils.prepare_archival_index_from_files_compute_embeddings(self.archival_storage_files)
        if self.compute_embeddings and self.archival_storage_index:
            self.index, self.archival_database = utils.prepare_archival_index(self.archival_storage_index)
        else:
            self.archival_database = utils.prepare_archival_index_from_files(self.archival_storage_files)

    def to_dict(self):
        return {
            "model": self.model,
            "memgpt_persona": self.memgpt_persona,
            "human_persona": self.human_persona,
            "preload_archival": self.preload_archival,
            "archival_storage_files": self.archival_storage_files,
            "archival_storage_index": self.archival_storage_index,
            "compute_embeddings": self.compute_embeddings,
            "load_type": self.load_type,
            "agent_save_file": self.agent_save_file,
            "persistence_manager_save_file": self.persistence_manager_save_file,
            "host": self.host,
        }

    def load_config(self, config_file):
        with open(config_file, "rt") as f:
            cfg = json.load(f)
        self.model = cfg["model"]
        self.memgpt_persona = cfg["memgpt_persona"]
        self.human_persona = cfg["human_persona"]
        self.preload_archival = cfg["preload_archival"]
        self.archival_storage_files = cfg["archival_storage_files"]
        self.archival_storage_index = cfg["archival_storage_index"]
        self.compute_embeddings = cfg["compute_embeddings"]
        self.load_type = cfg["load_type"]
        self.agent_save_file = cfg["agent_save_file"]
        self.persistence_manager_save_file = cfg["persistence_manager_save_file"]
        self.host = cfg["host"]

    def write_config(self, configs_dir=None):
        if configs_dir is None:
            configs_dir = Config.configs_dir
        os.makedirs(configs_dir, exist_ok=True)
        if self.config_file is None:
            filename = os.path.join(configs_dir, utils.get_local_time().replace(" ", "_").replace(":", "_"))
            self.config_file = f"{filename}.json"
        with open(self.config_file, "wt") as f:
            json.dump(self.to_dict(), f, indent=4)
        print(f"{Style.BRIGHT}{Fore.MAGENTA}⚙️ Saved config file to {self.config_file}.{Style.RESET_ALL}")

    @staticmethod
    def is_valid_config_file(file: str):
        cfg = Config()
        try:
            cfg.load_config(file)
        except Exception:
            return False
        return cfg.memgpt_persona is not None and cfg.human_persona is not None  # TODO: more validation for configs

    @staticmethod
    def get_memgpt_personas():
        dir_path = Config.personas_dir
        all_personas = Config.get_personas(dir_path)
        default_personas = [
            "sam",
            "sam_pov",
            "memgpt_starter",
            "memgpt_doc",
            "sam_simple_pov_gpt35",
        ]
        custom_personas_in_examples = list(set(all_personas) - set(default_personas))
        custom_personas = Config.get_personas(Config.custom_personas_dir)
        return (
            Config.get_persona_choices(
                [p for p in custom_personas],
                get_persona_text,
                Config.custom_personas_dir,
            )
            + Config.get_persona_choices(
                [p for p in custom_personas_in_examples + default_personas],
                get_persona_text,
                None,
                # Config.personas_dir,
            )
            + [
                questionary.Separator(),
                questionary.Choice(
                    f"📝 You can create your own personas by adding .txt files to {Config.custom_personas_dir}.",
                    disabled=True,
                ),
            ]
        )

    @staticmethod
    def get_user_personas():
        dir_path = Config.humans_dir
        all_personas = Config.get_personas(dir_path)
        default_personas = ["basic", "cs_phd"]
        custom_personas_in_examples = list(set(all_personas) - set(default_personas))
        custom_personas = Config.get_personas(Config.custom_humans_dir)
        return (
            Config.get_persona_choices(
                [p for p in custom_personas],
                get_human_text,
                Config.custom_humans_dir,
            )
            + Config.get_persona_choices(
                [p for p in custom_personas_in_examples + default_personas],
                get_human_text,
                None,
                # Config.humans_dir,
            )
            + [
                questionary.Separator(),
                questionary.Choice(
                    f"📝 You can create your own human profiles by adding .txt files to {Config.custom_humans_dir}.",
                    disabled=True,
                ),
            ]
        )

    @staticmethod
    def get_personas(dir_path) -> List[str]:
        files = sorted(glob.glob(os.path.join(dir_path, "*.txt")))
        stems = []
        for f in files:
            filename = os.path.basename(f)
            stem, _ = os.path.splitext(filename)
            stems.append(stem)
        return stems

    @staticmethod
    def get_persona_choices(personas, text_getter, dir):
        return [
            questionary.Choice(
                title=[
                    ("class:question", f"{p}"),
                    ("class:text", f"\n{indent(text_getter(p, dir))}"),
                ],
                value=(p, dir),
            )
            for p in personas
        ]

    @staticmethod
    def get_most_recent_config(configs_dir=None):
        if configs_dir is None:
            configs_dir = Config.configs_dir
        os.makedirs(configs_dir, exist_ok=True)
        files = [
            os.path.join(configs_dir, f)
            for f in os.listdir(configs_dir)
            if os.path.isfile(os.path.join(configs_dir, f)) and Config.is_valid_config_file(os.path.join(configs_dir, f))
        ]
        # Return the file with the most recent modification time
        if len(files) == 0:
            return None
        return max(files, key=os.path.getmtime)


def indent(text, num_lines=5):
    lines = textwrap.fill(text, width=100).split("\n")
    if len(lines) > num_lines:
        lines = lines[: num_lines - 1] + ["... (truncated)", lines[-1]]
    return "     " + "\n     ".join(lines)
