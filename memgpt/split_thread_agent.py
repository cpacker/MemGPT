import datetime

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

from memgpt.agent import Agent, save_agent
from memgpt.constants import (
    FIRST_MESSAGE_ATTEMPTS,
)
from memgpt.interface import AgentInterface
from memgpt.metadata import MetadataStore
from memgpt.schemas.agent import AgentState
from memgpt.schemas.enums import OptionState
from memgpt.schemas.memory import Memory
from memgpt.schemas.message import Message
from memgpt.schemas.tool import Tool


class AbstractAgent(ABC):
    """
    Abstract class for conversational agents.
    """

    agent_state: AgentState
    memory: Memory
    interface: AgentInterface

    @abstractmethod
    def step(
        self,
        user_message: Union[Message, str],  # NOTE: should be json.dump(dict)
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,  # if True, return dicts, if False, return Message objects
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
        stream: bool = False,  # TODO move to config?
        timestamp: Optional[datetime.datetime] = None,
        inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
        ms: Optional[MetadataStore] = None,
    ) -> Tuple[List[Union[dict, Message]], bool, bool, bool]:
        """
        Top-level event message handler for the agent.
        """
        pass


class SplitThreadAgent(AbstractAgent):
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

    def step(
        self,
        user_message: Union[Message, str],  # NOTE: should be json.dump(dict)
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,  # if True, return dicts, if False, return Message objects
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
        stream: bool = False,  # TODO move to config?
        timestamp: Optional[datetime.datetime] = None,
        inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
        ms: Optional[MetadataStore] = None,
    ) -> Tuple[List[Union[dict, Message]], bool, bool, bool]:
        memory_step = self.memory_agent.step(
            user_message=user_message,
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

        for i in memory_step[0]:
            if not i.text:
                continue
            i.text = f"MEMORY: {i.text}"

        conversation_step = self.conversation_agent.step(
            user_message=user_message,
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
        print("I GOT CONVERSATION STEP", conversation_step)

        combined_steps = (
            memory_step[0] + conversation_step[0],
            memory_step[1] or conversation_step[1],
            memory_step[2] or conversation_step[2],
            memory_step[3] or conversation_step[3],
            memory_step[4] + conversation_step[4],
        )
        return combined_steps

    def update_state(self) -> AgentState:
        self.conversation_agent.update_state()
        self.memory_agent.update_state()
        self.agent.update_state()
        return self.agent_state


def save_split_thread_agent(agent: SplitThreadAgent, ms: MetadataStore):
    """Save agent to metadata store"""

    assert agent.agent_state.split_thread_agent, "Agent state must be a split thread agent."

    # save conversational agent
    save_agent(agent=agent.agent, ms=ms)
    save_agent(agent=agent.conversation_agent, ms=ms)
    save_agent(agent=agent.memory_agent, ms=ms)
