import datetime
import inspect
import traceback

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Union
from tqdm import tqdm

from memgpt.agent import Agent, save_agent
from memgpt.agent_store.storage import StorageConnector
from memgpt.constants import (
    CLI_WARNING_PREFIX,
    FIRST_MESSAGE_ATTEMPTS,
    IN_CONTEXT_MEMORY_KEYWORD,
    LLM_MAX_TOKENS,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
    MESSAGE_SUMMARY_WARNING_FRAC,
)
from memgpt.interface import AgentInterface
from memgpt.llm_api.llm_api_tools import create, is_context_overflow_error
from memgpt.memory import ArchivalMemory, RecallMemory, summarize_messages
from memgpt.metadata import MetadataStore
from memgpt.persistence_manager import LocalStateManager
from memgpt.schemas.agent import AgentState
from memgpt.schemas.block import Block
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.enums import OptionState
from memgpt.schemas.memory import Memory
from memgpt.schemas.message import Message
from memgpt.schemas.openai.chat_completion_response import ChatCompletionResponse
from memgpt.schemas.openai.chat_completion_response import (
    Message as ChatCompletionMessage,
)
from memgpt.schemas.passage import Passage
from memgpt.schemas.tool import Tool
from memgpt.system import (
    get_initial_boot_messages,
    get_login_event,
    package_function_response,
    package_summarize_message,
)
from memgpt.utils import (
    count_tokens,
    get_local_time,
    get_tool_call_id,
    get_utc_time,
    is_utc_datetime,
    json_dumps,
    json_loads,
    parse_json,
    printd,
    united_diff,
    validate_function_response,
    verify_first_message_correctness,
)


class AbstractAgent(ABC):
    """
    Abstract class for conversational agents.
    """

    # agent_state: AgentState
    # memory: Memory
    # interface: AgentInterface

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

    # @abstractmethod
    # def update_state(self) -> AgentState:
    #     """
    #     Update the agent state.
    #     """
    #     pass

    # @property
    # @abstractmethod
    # def messages(self) -> List[dict]:
    #     pass

    # @messages.setter
    # @abstractmethod
    # def messages(self, value: List[dict]):
    #     pass


class SplitThreadAgent(AbstractAgent):
    def __init__(
        self,
        interface: AgentInterface,
        # agents can be created from providing agent_state
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
        self.conversational_agent = Agent(
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
        self.interface = interface

    @property
    def memory(self) -> Memory:
        return self.conversational_agent.memory

    @memory.setter
    def memory(self, value: Memory):
        self.conversational_agent.memory = value

    @property
    def agent_state(self) -> AgentState:
        return self.conversational_agent.agent_state

    @agent_state.setter
    def agent_state(self, value: AgentState):
        self.conversational_agent.agent_state = value

    # @property
    # def interface(self) -> AgentInterface:
    #     return self.conversational_agent.interface

    # @interface.setter
    # def interface(self, value: AgentInterface):
    #     self.conversational_agent.interface = value

    # @property
    # def messages(self) -> List[dict]:
    #     return self.conversational_agent.messages

    # @messages.setter
    # def messages(self, value: List[dict]):
    #     raise ValueError("Cannot set messages directly on SplitThreadAgent")

    # @property
    # def interface(self) -> AgentInterface:
    #     print("HELLLLOOO I AM BEING CALLED !!!")
    #     return self.conversational_agent.interface

    # @interface.setter
    # def interface(self, value: AgentInterface):
    #     self.conversational_agent.interface

    # def update_state(self) -> AgentState:
    #     return self.conversational_agent.update_state()
    #     # message_ids = [msg.id for msg in self.conversational_agent._messages]
    # assert isinstance(self.conversational_agent.memory, Memory), f"Memory is not a Memory object: {type(self.memory)}"

    # # override any fields that may have been updated
    # self.agent_state.message_ids = message_ids
    # self.agent_state.memory = self.convememory
    # self.agent_state.system = self.system

    # return self.agent_state

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
        return self.conversational_agent.step(
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

    def update_state(self) -> AgentState:
        return self.conversational_agent.update_state()


# SplitThreadAgent = Agent


def save_split_thread_agent(agent: SplitThreadAgent, ms: MetadataStore):
    """Save agent to metadata store"""

    # save conversational agent
    save_agent(agent=agent.conversational_agent, ms=ms)
    save_agent(agent=agent.memory_agent, ms=ms)

    # agent.update_state()
    # agent_state = agent.agent_state
    # agent_id = agent_state.id
    # assert isinstance(agent_state.memory, Memory), f"Memory is not a Memory object: {type(agent_state.memory)}"

    # # NOTE: we're saving agent memory before persisting the agent to ensure
    # # that allocated block_ids for each memory block are present in the agent model
    # save_agent_memory(agent=agent, ms=ms)

    # if ms.get_agent(agent_id=agent.agent_state.id):
    #     ms.update_agent(agent_state)
    # else:
    #     ms.create_agent(agent_state)

    # agent.agent_state = ms.get_agent(agent_id=agent_id)
    # assert isinstance(agent.agent_state.memory, Memory), f"Memory is not a Memory object: {type(agent_state.memory)}"
