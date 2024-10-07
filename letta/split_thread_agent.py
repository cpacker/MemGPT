import datetime

from typing import List, Optional, Tuple, Union

from letta.agent import BaseAgent, Agent, save_agent
from letta.constants import (
    FIRST_MESSAGE_ATTEMPTS,
)
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.schemas.agent import AgentState, AgentStepResponse
from letta.schemas.agent_config import AgentType
from letta.schemas.enums import OptionState
from letta.schemas.memory import Memory
from letta.schemas.message import Message
from letta.schemas.tool import Tool


class SplitThreadAgent(BaseAgent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        conversation_agent_state: AgentState,
        conversation_tools: List[Tool],
        memory_agent_state: AgentState,
        memory_tools: List[Tool],
        # memory: Memory,
        # extras
        messages_total: Optional[int] = None,  # TODO remove?
        first_message_verify_mono: bool = True,  # TODO move to config?
    ):
        self.conversation_agent = Agent(
            interface=interface,
            agent_state=conversation_agent_state,
            tools=conversation_tools,
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )
        self.memory_agent = Agent(
            interface=interface,
            agent_state=memory_agent_state,
            tools=memory_tools,
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )
        self.agent = Agent(
            interface=interface,
            agent_state=agent_state,
            tools=conversation_tools + memory_tools,
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )
        self.interface = interface

    def step(
        self,
        messages: Union[Message, List[Message], str],  # TODO deprecate str inputs
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,  # if True, return dicts, if False, return Message objects
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
        stream: bool = False,  # TODO move to config?
        timestamp: Optional[datetime.datetime] = None,
        inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
        ms: Optional[MetadataStore] = None,
    ) -> AgentStepResponse:
        memory_step = self.memory_agent.step(
            user_message=messages,
            first_message=first_message,
            first_message_retry_limit=first_message_retry_limit,
            skip_verify=skip_verify,
            return_dicts=return_dicts,
            recreate_message_timestamp=recreate_message_timestamp,
            stream=stream,
            timestamp=timestamp,
            inner_thoughts_in_kwargs=inner_thoughts_in_kwargs,
            ms=ms,
        )

        conversation_step = self.conversation_agent.step(
            user_message=messages,
            first_message=first_message,
            first_message_retry_limit=first_message_retry_limit,
            skip_verify=skip_verify,
            return_dicts=return_dicts,
            recreate_message_timestamp=recreate_message_timestamp,
            stream=stream,
            timestamp=timestamp,
            inner_thoughts_in_kwargs=inner_thoughts_in_kwargs,
            ms=ms,
        )

        combined_step = AgentStepResponse(
            messages=memory_step.messages + conversation_step.messages,
            heartbeat_request=memory_step.heartbeat_request or conversation_step.heartbeat_request,
            function_failed=memory_step.function_failed or conversation_step.function_failed,
            in_context_memory_warning=memory_step.in_context_memory_warning or conversation_step.in_context_memory_warning,
            usage=memory_step.usage + conversation_step.usage,
        )
        return combined_step

    def update_state(self) -> AgentState:
        self.conversation_agent.update_state()
        self.memory_agent.update_state()
        self.agent.update_state()
        return self.agent_state

    @property
    def agent_state(self) -> AgentState:
        return self.agent.agent_state

    @agent_state.setter
    def agent_state(self, value: AgentState):
        self.agent.agent_state = value

    @property
    def memory(self) -> Memory:
        return self.agent.memory

    @memory.setter
    def memory(self, value: Memory):
        self.agent.memory = value


def save_split_thread_agent(agent: SplitThreadAgent, ms: MetadataStore):
    """Save agent to metadata store"""

    assert agent.agent_state.agent_config.agent_type == AgentType.split_thread_agent, "Agent state must be a split thread agent."

    # save conversational agent
    save_agent(agent=agent.agent, ms=ms)
    save_agent(agent=agent.conversation_agent, ms=ms)
    save_agent(agent=agent.memory_agent, ms=ms)
