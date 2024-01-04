import os
from typing import Dict, List, Union

from memgpt.cli.cli import QuickstartChoice
from memgpt.cli.cli import set_config_with_dict, quickstart as quickstart_func, str_to_quickstart_choice
from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.persistence_manager import PersistenceManager
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer


class Client(object):
    def __init__(
        self,
        auto_save: bool = False,
        quickstart: Union[QuickstartChoice, str, None] = None,
        config: Union[Dict, MemGPTConfig] = None,  # not the same thing as AgentConfig
        debug: bool = False,
    ):
        """
        Initializes a new instance of Client class.
        :param auto_save: indicates whether to automatically save after every message.
        :param quickstart: allows running quickstart on client init.
        :param config: optional config settings to apply after quickstart
        :param debug: indicates whether to display debug messages.
        """
        self.user_id = "null"
        self.auto_save = auto_save

        # make sure everything is set up properly
        MemGPTConfig.create_config_dir()

        # If this is the first ever start, do basic initialization
        if not MemGPTConfig.exists() and config is None and quickstart is None:
            # Default to openai
            print("Detecting uninitialized MemGPT, defaulting to quickstart == openai")
            quickstart = "openai"

        if quickstart:
            # api key passed in config has priority over env var
            if isinstance(config, dict) and "openai_api_key" in config:
                openai_key = config["openai_api_key"]
            else:
                openai_key = os.environ.get("OPENAI_API_KEY", None)

            # throw an error if we can't resolve the key
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            elif quickstart == QuickstartChoice.openai or quickstart == "openai":
                raise ValueError("Please set OPENAI_API_KEY or pass 'openai_api_key' in config dict")

            if isinstance(quickstart, str):
                quickstart = str_to_quickstart_choice(quickstart)
            quickstart_func(backend=quickstart, debug=debug)

        if config is not None:
            set_config_with_dict(config)

        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface=self.interface)

    def list_agents(self):
        self.interface.clear()
        return self.server.list_agents(user_id=self.user_id)

    def agent_exists(self, agent_id: str) -> bool:
        existing = self.list_agents()
        return agent_id in existing["agent_names"]

    def create_agent(
        self,
        agent_config: Union[Dict, AgentConfig],
        persistence_manager: Union[PersistenceManager, None] = None,
        throw_if_exists: bool = False,
    ) -> str:
        if isinstance(agent_config, dict):
            agent_name = agent_config.get("name")
        else:
            agent_name = agent_config.name

        if not self.agent_exists(agent_id=agent_name):
            self.interface.clear()
            return self.server.create_agent(user_id=self.user_id, agent_config=agent_config, persistence_manager=persistence_manager)

        if throw_if_exists:
            raise ValueError(f"Agent {agent_name} already exists")

        return agent_name

    def get_agent_config(self, agent_id: str) -> Dict:
        self.interface.clear()
        return self.server.get_agent_config(user_id=self.user_id, agent_id=agent_id)

    def get_agent_memory(self, agent_id: str) -> Dict:
        self.interface.clear()
        return self.server.get_agent_memory(user_id=self.user_id, agent_id=agent_id)

    def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Dict:
        self.interface.clear()
        return self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)

    def user_message(self, agent_id: str, message: str) -> List[Dict]:
        self.interface.clear()
        self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()
        return self.interface.to_list()

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        self.interface.clear()
        return self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

    def save(self):
        self.server.save_agents()
