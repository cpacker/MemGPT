from typing import List, Optional, Union

from letta.agent import Agent
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.schemas.agent import AgentState
from letta.schemas.message import Message
from letta.schemas.tool import Tool
from letta.schemas.usage import LettaUsageStatistics


def send_message_offline_agent(self: "Agent", message: str) -> Optional[str]:
    """
    Sends a message to the human user. The function is the same as the base send_message function, but is used
    when we do not include the other base tools.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    # FIXME passing of msg_obj here is a hack, unclear if guaranteed to be the correct reference
    self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    return None


def trigger_rethink_memory(self: "Agent", message: Optional[str]) -> Optional[str]:
    """
    If the user says the word "rethink_memory" then call this function

    Args:
        message (Optional[str]): String message to condition what the memory agent should rethink about.

    """
    from letta import create_client

    client = create_client()
    agents = client.list_agents()
    for agent in agents:
        if agent.agent_type == "offline_memory_agent":
            client.user_message(agent_id=agent.id, message=message)


def rethink_memory(self: "Agent") -> Optional[str]:
    """
    Goes through the memory and rethinks the memory based on the new message.

    """
    for memory_block in self.memory:
        print(memory_block.value)
    self.memory.update_block_value(name="rethink_memory_block", value="Rethinking memory")
    return None


class OfflineMemoryAgent(Agent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        tools: List[Tool] = [],
        first_message_verify_mono: bool = False,
    ):
        super().__init__(interface, agent_state, tools)
        self.tools = tools
        self.first_message_verify_mono = first_message_verify_mono

    def step(
        self,
        messages: Union[Message, List[Message]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        ms: Optional[MetadataStore] = None,
        **kwargs,
    ) -> LettaUsageStatistics:
        """Go through what is currently in memory core memory and information."""
        return super().step(messages, chaining, max_chaining_steps, ms, **kwargs)
