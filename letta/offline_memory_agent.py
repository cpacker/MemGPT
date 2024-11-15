import json
from typing import List, Optional, Union

from letta.agent import Agent, save_agent
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.schemas.agent import AgentState
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import UsageStatistics
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


def rethink_memory(self, new_memory: str, block_label: Optional[str]) -> Optional[str]:
    """
    Rethinks the memory in the memory block titled `block_name`,
    and integrates the information from the memory block into the memory.

    Args:
        new_memory (str): The new memory with information integrated from the memory block.
        block_label (str): The name of the block to integrate information from. None all the information has been integrated to terminate the loop.
    """

    self.memory.update_block_value(label="rethink_memory_block", value=new_memory)
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
        """Go through what is currently in memory core memory and integrate information."""

        next_input_message = messages if isinstance(messages, list) else [messages]
        counter = 0
        total_usage = UsageStatistics()
        step_count = 0

        current_block_label = "rethink_memory_block"
        while current_block_label in self.memory.list_block_labels():
            kwargs["ms"] = ms
            kwargs["first_message"] = False
            step_response = self.inner_step(
                messages=next_input_message,
                **kwargs,
            )
            print(step_response)
            for message in step_response.messages:
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        arguments = json.loads(tool_call.function.arguments)
                        current_block_label = arguments["block_label"]

            usage = step_response.usage
            step_count += 1
            total_usage += usage
            counter += 1
            self.interface.step_complete()
            if ms:
                save_agent(self, ms)

        return LettaUsageStatistics(**total_usage.model_dump(), step_count=step_count)
