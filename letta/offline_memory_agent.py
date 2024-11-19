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
    Called if and only when user says the word "trigger_rethink_memory". It will trigger the re-evaluation of the memory.

    Args:
        message (Optional[str]): Description of what aspect of the memory should be re-evaluated.

    """
    from letta import create_client

    recent_convo = "".join([str(message) for message in self.messages])
    self.memory.update_block_value(label="conversation_block", value=recent_convo)

    client = create_client()
    agents = client.list_agents()
    for agent in agents:
        if agent.agent_type == "offline_memory_agent":
            response = client.user_message(agent_id=agent.id, message=message)
            print(response)


def rethink_memory(self, new_memory: str, target_block_label: Optional[str], source_block_label: Optional[str]) -> Optional[str]:
    """
    Re-evaluate the memory in block_name, integrating new and updated facts.
    Replace outdated information with the most likely truths, avoiding redundancy with original memories.
    Ensure consistency with other memory blocks.

    Args:
        new_memory (str): The new memory with information integrated from the memory block.
        source_block_label (str): The name of the block to integrate information from. None if all the information has been integrated to terminate the loop.
        target_block_label (str): The name of the block to write to.
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    if target_block_label is not None:
        if self.memory.get_block(target_block_label) is None:
            self.memory.create_block(label=target_block_label, value=new_memory)
        self.memory.update_block_value(label=target_block_label, value=new_memory)

    print(f"Rethinking memory for block {target_block_label} with new memory: {new_memory} from block {source_block_label}")
    return None


def finish_rethinking_memory(self) -> Optional[str]:
    """
    This function is called when the agent is done rethinking the memory.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None


class OfflineMemoryAgent(Agent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        tools: List[Tool] = [],
        first_message_verify_mono: bool = False,
        max_memory_rethinks: int = 10,
    ):
        super().__init__(interface, agent_state, tools)
        self.tools = tools
        self.first_message_verify_mono = first_message_verify_mono
        self.max_memory_rethinks = max_memory_rethinks

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

        while counter < self.max_memory_rethinks:
            kwargs["ms"] = ms
            kwargs["first_message"] = False
            step_response = self.inner_step(
                messages=next_input_message,
                **kwargs,
            )
            for message in step_response.messages:
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        # check if the function name is "finish_rethinking_memory"
                        if tool_call.function.name == "finish_rethinking_memory":
                            counter = self.max_memory_rethinks
                            break
                        json.loads(tool_call.function.arguments)

            usage = step_response.usage
            step_count += 1
            total_usage += usage
            counter += 1
            self.interface.step_complete()

            if ms:
                save_agent(self, ms)

        return LettaUsageStatistics(**total_usage.model_dump(), step_count=step_count)
