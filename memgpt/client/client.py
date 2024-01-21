import os
import uuid
from typing import Dict, List, Union, Optional, Tuple

from memgpt.data_types import AgentState
from memgpt.cli.cli import QuickstartChoice
from memgpt.cli.cli import set_config_with_dict, quickstart as quickstart_func, str_to_quickstart_choice
from memgpt.config import MemGPTConfig
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer


class Client(object):
    def __init__(
        self,
        user_id: str = None,
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
        self.auto_save = auto_save

        # make sure everything is set up properly
        # TODO: remove this eventually? for multi-user, we can't have a shared config directory
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

        if user_id is None:
            # the default user_id
            config = MemGPTConfig.load()
            self.user_id = uuid.UUID(config.anon_clientid)
        elif isinstance(user_id, str):
            self.user_id = uuid.UUID(user_id)
        elif isinstance(user_id, uuid.UUID):
            self.user_id = user_id
        else:
            raise TypeError(user_id)
        self.interface = QueuingInterface(debug=debug)
        self.server = SyncServer(default_interface=self.interface)

    def list_agents(self):
        self.interface.clear()
        return self.server.list_agents(user_id=self.user_id)

    def agent_exists(self, agent_id: Optional[str] = None, agent_name: Optional[str] = None) -> bool:
        if not (agent_id or agent_name):
            raise ValueError(f"Either agent_id or agent_name must be provided")
        if agent_id and agent_name:
            raise ValueError(f"Only one of agent_id or agent_name can be provided")
        existing = self.list_agents()
        if agent_id:
            return agent_id in [agent["id"] for agent in existing["agents"]]
        else:
            return agent_name in [agent["name"] for agent in existing["agents"]]

    def create_agent(
        self,
        agent_config: dict,
    ) -> AgentState:
        if isinstance(agent_config, dict):
            agent_name = agent_config.get("name")
        else:
            raise TypeError(f"agent_config must be of type dict")

        if "name" in agent_config and self.agent_exists(agent_name=agent_config["name"]):
            raise ValueError(f"Agent with name {agent_config['name']} already exists (user_id={self.user_id})")

        self.interface.clear()
        agent_state = self.server.create_agent(user_id=self.user_id, agent_config=agent_config)
        return agent_state

    def get_agent_config(self, agent_id: str) -> Dict:
        self.interface.clear()
        return self.server.get_agent_config(user_id=self.user_id, agent_id=agent_id)

    def get_agent_memory(self, agent_id: str) -> Dict:
        self.interface.clear()
        return self.server.get_agent_memory(user_id=self.user_id, agent_id=agent_id)

    def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Dict:
        self.interface.clear()
        return self.server.update_agent_core_memory(user_id=self.user_id, agent_id=agent_id, new_memory_contents=new_memory_contents)

    def user_message(self, agent_id: str, message: str, return_token_count: bool = False) -> Union[List[Dict], Tuple[List[Dict], int]]:
        self.interface.clear()
        tokens_accumulated = self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()
        if return_token_count:
            return self.interface.to_list(), tokens_accumulated
        else:
            return self.interface.to_list()

    def run_command(self, agent_id: str, command: str) -> Union[str, None]:
        self.interface.clear()
        return self.server.run_command(user_id=self.user_id, agent_id=agent_id, command=command)

    def save(self):
        self.server.save_agents()
