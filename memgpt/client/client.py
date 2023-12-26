from typing import Union

from memgpt.config import AgentConfig
from memgpt.interface import AgentInterface
from memgpt.interface import CLIInterface
from memgpt.persistence_manager import PersistenceManager
from memgpt.server.server import SyncServer


class Client(object):

    def __init__(
        self,
        interface: AgentInterface = CLIInterface(),
        auto_save: bool = False,
    ):
        self.user_id = "null"
        self.auto_save = auto_save
        self.server = SyncServer(default_interface=interface)

    def create_agent(
        self,
        agent_config: Union[dict, AgentConfig],
        persistence_manager: Union[PersistenceManager, None] = None,
    ) -> str:
        return self.server.create_agent(
            user_id=self.user_id,
            agent_config=agent_config,
            persistence_manager=persistence_manager)

    def user_message(self, agent_id: str, message: str):
        self.server.user_message(user_id=self.user_id, agent_id=agent_id, message=message)
        if self.auto_save:
            self.save()

    def save(self):
        self.server.save_agents()


if __name__ == "__main__":
    from memgpt.persistence_manager import LocalStateManager
    import memgpt.constants as constants
    import memgpt.system as system
    import memgpt.utils as utils
    import memgpt.presets.presets as presets
    from memgpt.config import AgentConfig
    from memgpt.interface import CLIInterface as interface
    from memgpt.constants import MEMGPT_DIR

    # Create an AgentConfig option from the inputs
    # agent_config = AgentConfig(
    #     # name="agent_4",
    #     name="tommy",
    #     persona=constants.DEFAULT_PERSONA,
    #     human=constants.DEFAULT_HUMAN,
    #     preset="memgpt_chat",
    #     model="gpt-4",
    #     model_endpoint_type="openai",
    #     model_endpoint="https://api.openai.com/v1",
    #     context_window=8192,
    # )
    #
    # client = Client(auto_save=True)
    # name = client.create_agent(agent_config=agent_config)
    # print(name)


    #client.user_message(agent_id="agent_5", message="tell me my name please")
    #client.user_message(agent_id="agent_5", message="I enjoy dry red wine")
    #client.user_message(agent_id="agent_5", message="I have a dog named Tonic")
    #client.save()
    #client.user_message(agent_id="agent_5", message="What is my dog's name?")

