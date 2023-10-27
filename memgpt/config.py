import glob
import json
import os
import textwrap


import questionary

from colorama import Fore, Style

from typing import List, Type

import memgpt.utils as utils
import memgpt.interface as interface
from memgpt.personas.personas import get_persona_text
from memgpt.humans.humans import get_human_text
from memgpt.constants import MEMGPT_DIR

model_choices = [
    questionary.Choice("gpt-4"),
    questionary.Choice(
        "gpt-3.5-turbo (experimental! function-calling performance is not quite at the level of gpt-4 yet)",
        value="gpt-3.5-turbo",
    ),
]


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
    async def legacy_flags_init(
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
            await self.configure_archival_storage(recompute_embeddings)
        return self

    @classmethod
    async def config_init(cls: Type["Config"], config_file: str = None):
        self = cls()
        self.config_file = config_file
        if self.config_file is None:
            cfg = Config.get_most_recent_config()
            use_cfg = False
            if cfg:
                print(f"{Style.BRIGHT}{Fore.MAGENTA}⚙️ Found saved config file.{Style.RESET_ALL}")
                use_cfg = await questionary.confirm(f"Use most recent config file '{cfg}'?").ask_async()
            if use_cfg:
                self.config_file = cfg

        if self.config_file:
            self.load_config(self.config_file)
            recompute_embeddings = False
            if self.compute_embeddings:
                if self.archival_storage_index:
                    recompute_embeddings = await questionary.confirm(
                        f"Would you like to recompute embeddings? Do this if your files have changed.\n    Files: {self.archival_storage_files}",
                        default=False,
                    ).ask_async()
                else:
                    recompute_embeddings = True
            if self.load_type:
                await self.configure_archival_storage(recompute_embeddings)
                self.write_config()
            return self

        # print("No settings file found, configuring MemGPT...")
        print(f"{Style.BRIGHT}{Fore.MAGENTA}⚙️ No settings file found, configuring MemGPT...{Style.RESET_ALL}")

        self.model = await questionary.select(
            "Which model would you like to use?",
            model_choices,
            default=model_choices[0],
        ).ask_async()

        self.memgpt_persona = await questionary.select(
            "Which persona would you like MemGPT to use?",
            Config.get_memgpt_personas(),
        ).ask_async()
        print(self.memgpt_persona)

        self.human_persona = await questionary.select(
            "Which user would you like to use?",
            Config.get_user_personas(),
        ).ask_async()

        self.archival_storage_index = None
        self.preload_archival = await questionary.confirm("Would you like to preload anything into MemGPT's archival memory?", default=False).ask_async()
        if self.preload_archival:
            self.load_type = await questionary.select(
                "What would you like to load?",
                choices=[
                    questionary.Choice("A folder or file", value="folder"),
                    questionary.Choice("A SQL database", value="sql"),
                    questionary.Choice("A glob pattern", value="glob"),
                ],
            ).ask_async()
            if self.load_type == "folder" or self.load_type == "sql":
                archival_storage_path = await questionary.path("Please enter the folder or file (tab for autocomplete):").ask_async()
                if os.path.isdir(archival_storage_path):
                    self.archival_storage_files = os.path.join(archival_storage_path, "*")
                else:
                    self.archival_storage_files = archival_storage_path
            else:
                self.archival_storage_files = await questionary.path("Please enter the glob pattern (tab for autocomplete):").ask_async()
            self.compute_embeddings = await questionary.confirm(
                "Would you like to compute embeddings over these files to enable embeddings search?"
            ).ask_async()
            await self.configure_archival_storage(self.compute_embeddings)

        self.write_config()
        return self

    async def configure_archival_storage(self, recompute_embeddings):
        if recompute_embeddings:
            if self.host:
                interface.warning_message(
                    "⛔️ Embeddings on a non-OpenAI endpoint are not yet supported, falling back to substring matching search."
                )
            else:
                self.archival_storage_index = await utils.prepare_archival_index_from_files_compute_embeddings(self.archival_storage_files)
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
