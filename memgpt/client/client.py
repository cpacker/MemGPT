from typing import Dict, List, Union

from memgpt.config import AgentConfig
from memgpt.persistence_manager import PersistenceManager
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer


class Client(object):

    def __init__(
        self,
        auto_save: bool = False,
    ):
        self.user_id = "null"
        self.auto_save = auto_save
        self.interface = QueuingInterface()
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
        throw_if_exists: bool = False
    ) -> str:

        if not self.agent_exists(agent_id=agent_config.name):
            self.interface.clear()
            return self.server.create_agent(
                user_id=self.user_id,
                agent_config=agent_config,
                persistence_manager=persistence_manager)

        if throw_if_exists:
            raise ValueError(f"Agent {agent_config.name} already exists")

        return agent_config.name

    def get_agent_config(self, agent_id: str) -> Dict:
        self.interface.clear()
        return self.server.get_agent_config(user_id=self.user_id, agent_id=agent_id)

    def get_agent_memory(self, agent_id: str) -> Dict:
        self.interface.clear()
        return self.server.get_agent_memory(user_id=self.user_id, agent_id=agent_id)

    def update_agent_core_memory(self, agent_id: str, new_memory_contents: Dict) -> Dict:
        self.interface.clear()
        return self.server.update_agent_core_memory(
            user_id=self.user_id,
            agent_id=agent_id,
            new_memory_contents=new_memory_contents
        )

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
        self.interface.clear()
        self.server.save_agents()
