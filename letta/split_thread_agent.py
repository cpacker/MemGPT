import datetime

from typing import List, Optional, Tuple, Union

from letta.schemas.agent_config import AgentConfig, AgentType
from letta.schemas.agent import AgentState, CreateAgent
from letta.agent import BaseAgent, Agent, save_agent
from letta.constants import (
    FIRST_MESSAGE_ATTEMPTS,
)
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.prompts import gpt_system
from letta.schemas.agent import AgentState, AgentStepResponse
from letta.schemas.agent_config import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import OptionState
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.tool import Tool

from IPython import embed

MEMORY_TOOLS = [
    "core_memory_append",
    "core_memory_replace",
    "archival_memory_insert",
]


class SplitThreadAgent(BaseAgent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        conversation_agent_state: AgentState,
        conversation_tools: List[Tool],
        memory_agent_state: AgentState,
        memory_tools: List[Tool],
        # extras
        messages_total: Optional[int] = None,  # TODO remove?
        first_message_verify_mono: bool = True,  # TODO move to config?
    ):
        self.agent_state = agent_state
        self.memory = agent_state.memory
        self.system = agent_state.system
        self.interface = interface

        self.agent = Agent(
            interface=interface,
            agent_state=agent_state,
            tools=[],
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )

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

        self.update_state()

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

        self.agent_state = self.agent.agent_state
        self.agent_state.memory = self.memory
        self.agent_state.system = self.system

        return self.agent_state


def create_split_thread_agent(
    request: CreateAgent,
    user_id: str,
    tool_objs: List[Tool],
    agent_config: AgentConfig,
    llm_config: LLMConfig,
    embedding_config: EmbeddingConfig,
    interface: AgentInterface,
) -> Tuple[SplitThreadAgent, AgentState]:
    conversation_prompt = gpt_system.get_system_text("split_conversation")
    memory_prompt = gpt_system.get_system_text("split_memory")

    memory_tool_objs = [i for i in tool_objs if i.name in MEMORY_TOOLS]
    conversation_tool_objs = [i for i in tool_objs if i.name not in MEMORY_TOOLS]

    conversation_agent_state = AgentState(
        name=f"{request.name}_conversation",
        user_id=user_id,
        tools=[i.name for i in conversation_tool_objs],
        agent_config=AgentConfig(agent_type=AgentType.base_agent),
        llm_config=llm_config,
        embedding_config=embedding_config,
        system=conversation_prompt,
        memory=request.memory,
        description=request.description,
        metadata_=request.metadata_,
    )

    memory_agent_state = AgentState(
        name=f"{request.name}_memory",
        user_id=user_id,
        tools=[i.name for i in memory_tool_objs],
        agent_config=AgentConfig(agent_type=AgentType.base_agent),
        llm_config=llm_config,
        embedding_config=embedding_config,
        system=memory_prompt,
        memory=request.memory,
        description=request.description,
        metadata_=request.metadata_,
    )

    agent_state = AgentState(
        name=request.name,
        user_id=user_id,
        tools=[],
        agent_config=agent_config,
        llm_config=llm_config,
        embedding_config=embedding_config,
        system=request.system,
        memory=request.memory,
        description=request.description,
        metadata_=request.metadata_,
    )

    agent = SplitThreadAgent(
        interface=interface,
        agent_state=agent_state,
        conversation_agent_state=conversation_agent_state,
        conversation_tools=conversation_tool_objs,
        memory_agent_state=memory_agent_state,
        memory_tools=memory_tool_objs,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=True if (llm_config.model is not None and "gpt-4" in llm_config.model) else False,
    )

    return agent, agent_state


def save_split_thread_agent(agent: SplitThreadAgent, ms: MetadataStore):
    """Save agent to metadata store"""

    assert agent.agent_state.agent_config.agent_type == AgentType.split_thread_agent, "Agent state must be a split thread agent."

    # save conversational agent
    save_agent(agent=agent.agent, ms=ms)
    save_agent(agent=agent.conversation_agent, ms=ms)
    save_agent(agent=agent.memory_agent, ms=ms)
    agent.update_state()
