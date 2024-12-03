from typing import List, Optional, Union

from letta.agent import Agent, save_agent
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.orm import User
from letta.schemas.agent import AgentState
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import UsageStatistics
from letta.schemas.tool import Tool
from letta.schemas.usage import LettaUsageStatistics


def trigger_rethink_memory(agent_state: "AgentState", message: Optional[str]) -> Optional[str]:
    """
    Called if and only when user says the word "trigger_rethink_memory". It will trigger the re-evaluation of the memory.

    Args:
        message (Optional[str]): Description of what aspect of the memory should be re-evaluated.

    """
    from letta import create_client

    client = create_client()
    agents = client.list_agents()
    for agent in agents:
        if agent.agent_type == "offline_memory_agent":
            # client.get_agent(agent.id)
            client.user_message(agent_id=agent.id, message=message)


def trigger_rethink_memory_convo(agent_state: "AgentState", message: Optional[str]) -> Optional[str]:
    """
    Called if and only when user says the word "trigger_rethink_memory". It will trigger the re-evaluation of the memory.

    Args:
        message (Optional[str]): Description of what aspect of the memory should be re-evaluated.

    """
    from letta import create_client

    client = create_client()
    recent_convo = "".join([str(message) for message in agent_state.messages])[-2000:]  # TODO: make a better representation of the convo history
    agent_state.memory.update_block_value(label="conversation_block", value=recent_convo)
    client.update_block(agent_state.memory.get_block("conversation_block").id, text=recent_convo)
    # client.update_agent(agent_id=agent_state.agent_state.id, memory=agent_state.memory)

    client = create_client()
    agents = client.list_agents()
    for agent in agents:
        if agent.agent_type == "offline_memory_agent":
            client.get_agent(agent.id)
            client.user_message(agent_id=agent.id, message=message)


def rethink_memory_convo(agent_state: "AgentState", new_memory: str, target_block_label: Optional[str], source_block_label: Optional[str]) -> Optional[str]:
    """
    Re-evaluate the memory in block_name, integrating new and updated facts.
    Replace outdated information with the most likely truths, avoiding redundancy with original memories.
    Ensure consistency with other memory blocks.

    Args:
        new_memory (str): The new memory with information integrated from the memory block. If there is no new information, then this should be the same as the content in the source block.
        source_block_label (str): The name of the block to integrate information from. None if all the information has been integrated to terminate the loop. This can by any block.
        target_block_label (str): The name of the block to write to. This should be `chat_agent_human_new` or `chat_agent_persona_new`.
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    # from letta import create_client

    # client = create_client()
    if target_block_label is not None:
        if agent_state.memory.get_block(target_block_label) is None:
            agent_state.memory.create_block(label=target_block_label, value=new_memory)
        agent_state.memory.update_block_value(label=target_block_label, value=new_memory)
        # block_id = agent_state.memory.get_block(target_block_label).id
        # client.update_block(block_id, text=new_memory)
        # client.update_agent(agent_id=self.agent_state.id, memory=self.agent_state.memory)
        # _ = client.get_agent(self.agent_state.id)

    print(f"Rethinking memory for block {target_block_label} with new memory: {new_memory} from block {source_block_label}")
    return None


def rethink_memory(agent_state: "AgentState", new_memory: str, target_block_label: Optional[str], source_block_label: Optional[str]) -> Optional[str]:
    """
    Re-evaluate the memory in block_name, integrating new and updated facts.
    Replace outdated information with the most likely truths, avoiding redundancy with original memories.
    Ensure consistency with other memory blocks.

    Args:
        new_memory (str): The new memory with information integrated from the memory block. If there is no new information, then this should be the same as the content in the source block.
        source_block_label (str): The name of the block to integrate information from. None if all the information has been integrated to terminate the loop.
        target_block_label (str): The name of the block to write to.
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    from letta import create_client

    client = create_client()
    if target_block_label is not None:
        if agent_state.memory.get_block(target_block_label) is None:
            agent_state.memory.create_block(label=target_block_label, value=new_memory)
        agent_state.memory.update_block_value(label=target_block_label, value=new_memory)
        # block_id = agent_state.memory.get_block(target_block_label).id
        # client.update_block(block_id, text=new_memory)
        # client.update_agent(agent_id=agent_state.id, memory=self.agent_state.memory)

    print(f"Rethinking memory for block {target_block_label} with new memory: {new_memory} from block {source_block_label}")
    return None


def finish_rethinking_memory(agent_state: "AgentState") -> Optional[str]:
    """
    This function is called when the agent is done rethinking the memory.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    return None


def finish_rethinking_memory_convo(agent_state: "AgentState") -> Optional[str]:
    """
    This function is called when the agent is done rethinking the memory.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    from letta import create_client

    client = create_client()
    agents = client.list_agents()

    agent_state.memory.update_block_value("chat_agent_human", agent_state.memory.get_block("chat_agent_human_new").value)
    agent_state.memory.update_block_value("chat_agent_persona", agent_state.memory.get_block("chat_agent_persona_new").value)
    for agent in agents:
        if agent.name == "conversation_agent":
            agent.memory.update_block_value(label="chat_agent_human", value=agent_state.memory.get_block("chat_agent_human_new").value)
            agent.memory.update_block_value(label="chat_agent_persona", value=agent_state.memory.get_block("chat_agent_persona_new").value)

            chat_persona_block = agent.memory.get_block("chat_agent_persona")
            chat_human_block = agent.memory.get_block("chat_agent_human")
            client.update_block(chat_persona_block.id, text=agent_state.memory.get_block("chat_agent_persona_new").value)
            client.update_block(chat_human_block.id, text=agent_state.memory.get_block("chat_agent_human_new").value)
            agent = client.get_agent(agent.id)

    return None


class OfflineMemoryAgent(Agent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        tools: List[Tool] = [],
        first_message_verify_mono: bool = False,
        max_memory_rethinks: int = 10,
        user: User = None,
    ):
        super().__init__(interface, agent_state, tools, user)
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
            usage = step_response.usage
            step_count += 1
            total_usage += usage
            counter += 1
            self.interface.step_complete()

            if ms:
                save_agent(self, ms)

        return LettaUsageStatistics(**total_usage.model_dump(), step_count=step_count)
