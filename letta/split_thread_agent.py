import datetime
import threading
from typing import List, Optional, Tuple, Union

from letta.agent import Agent, BaseAgent, save_agent
from letta.constants import FIRST_MESSAGE_ATTEMPTS
from letta.functions.functions import parse_source_code
from letta.functions.schema_generator import generate_schema
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.prompts import gpt_system
from letta.schemas.agent import AgentState, AgentStepResponse, AgentType, CreateAgent
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import OptionState
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import UsageStatistics
from letta.schemas.tool import Tool

MEMORY_TOOLS = [
    "core_memory_append",
    "core_memory_replace",
    "archival_memory_insert",
]


class SplitThreadAgent(BaseAgent):
    """
    SplitThreadAgent is an agent that splits the conversation and memory into two separate agents.
    The memory agent is run in a separate thread asynchronously to the conversation agent. The
    conversation agent has the ability to wait for the memory agent to finish before continuing.
    """

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

        # Placeholder agent that represents the split thread agent
        self.agent = Agent(
            interface=interface,
            agent_state=agent_state,
            tools=[],
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )

        # Conversation agent
        self.conversation_agent = Agent(
            interface=interface,
            agent_state=conversation_agent_state,
            tools=conversation_tools,
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )

        # Flag to indicate if the conversation agent decided to wait for memory update
        self.conversation_waited = False

        # Lock to prevent the conversation agent from stepping while memory is updating
        self.conversation_agent_lock = threading.Lock()

        # Tool to wait for memory update
        memory_wait_tool = Tool(
            name="wait_for_memory_update",
            source_type="python",
            source_code=parse_source_code(self.wait_for_memory_update),
            json_schema=generate_schema(self.wait_for_memory_update),
            description="",
            module="",
            user_id=conversation_agent_state.user_id,
            tags=[],
        )
        conversation_agent_state.tools.append(memory_wait_tool.name)
        self.conversation_agent.link_tools(conversation_tools + [memory_wait_tool])
        self.conversation_agent.update_state()

        # Memory agent
        self.memory_agent = Agent(
            interface=interface,
            agent_state=memory_agent_state,
            tools=memory_tools,
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )

        # Queue to store all requested memory steps
        self.memory_queue = []

        # Result of the memory step
        self.memory_result = None

        # Condition variable to wake up memory thread when new memory step is added
        self.memory_condition = threading.Condition()

        # Condition variable to wake up conversation agent when memory step is finished
        self.conversation_condition = threading.Condition()

        # Thread that runs memory agent
        self.memory_thread = threading.Thread(target=self._memory_step, args=())
        self.memory_thread.start()

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
        inner_thoughts_in_kwargs_option: OptionState = OptionState.DEFAULT,
        ms: Optional[MetadataStore] = None,
    ) -> AgentStepResponse:
        kwargs = {
            "user_message": messages,
            "first_message": first_message,
            "first_message_retry_limit": first_message_retry_limit,
            "skip_verify": skip_verify,
            "return_dicts": return_dicts,
            "recreate_message_timestamp": recreate_message_timestamp,
            "stream": stream,
            "timestamp": timestamp,
            "inner_thoughts_in_kwargs_option": inner_thoughts_in_kwargs_option,
            "ms": ms,
        }

        # First, add the memory step to the queue and wake up the memory thread
        with self.memory_condition:
            self.memory_queue.append(kwargs)
            self.memory_condition.notify()

        with self.conversation_agent_lock:
            conversation_step = self.conversation_agent.step(**kwargs)

            last_message = conversation_step.messages[-1]
            waited = isinstance(last_message, Message) and last_message.name == "wait_for_memory_update"

            if waited:
                # If the conversation agent decided to wait for memory update, we need a response after the memory update
                with self.conversation_condition:
                    while not self.memory_result:
                        self.conversation_condition.wait()
                    assert self.memory_result is not None, "Memory result should not be None after waiting for memory update"

                    # Flush the memory output into this step
                    conversation_step = self._combine_steps(conversation_step, self.memory_result)
                    self.memory_result = None

                next_conversation_step = self.conversation_agent.step(**kwargs)
                conversation_step = self._combine_steps(conversation_step, next_conversation_step)
            else:
                # If the conversation agent did not wait for memory update, we can flush the memory result
                with self.conversation_condition:
                    if self.memory_result:
                        # Flush the memory output into this step
                        conversation_step = self._combine_steps(self.memory_result, conversation_step)
                        self.memory_result = None

        return conversation_step

    def _memory_step(self) -> AgentStepResponse:
        """
        Function that runs the memory agent in a separate thread.
        """
        while True:
            # Wait for a new memory step to be added
            with self.memory_condition:
                while not self.memory_queue:
                    self.memory_condition.wait()

                kwargs = self.memory_queue.pop(0)
                memory_step = self.memory_agent.step(**kwargs)

            with self.conversation_condition:
                if self.memory_result:
                    # If we had a memory result from a previous step, combine it with the current memory step
                    self.memory_result = self._combine_steps(self.memory_result, memory_step)
                else:
                    self.memory_result = memory_step

                # Notify the conversation agent that the memory step is finished
                if not self.memory_queue:
                    self.conversation_condition.notify()

    def wait_for_memory_update(self):
        pass

    def _combine_steps(self, *steps: AgentStepResponse) -> AgentStepResponse:
        combined_step = AgentStepResponse(
            messages=[],
            heartbeat_request=False,
            function_failed=False,
            in_context_memory_warning=False,
            usage=UsageStatistics(),
        )

        for step in steps:
            combined_step.messages += step.messages
            combined_step.heartbeat_request = combined_step.heartbeat_request or step.heartbeat_request
            combined_step.function_failed = combined_step.function_failed or step.function_failed
            combined_step.in_context_memory_warning = combined_step.in_context_memory_warning or step.in_context_memory_warning
            combined_step.usage += step.usage

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
        agent_type=AgentType.memgpt_agent,
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
        agent_type=AgentType.memgpt_agent,
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
        agent_type=AgentType.split_thread_agent,
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

    assert agent.agent_state.agent_type == AgentType.split_thread_agent, "Agent state must be a split thread agent."

    # save conversational agent
    save_agent(agent=agent.agent, ms=ms)
    save_agent(agent=agent.conversation_agent, ms=ms)
    save_agent(agent=agent.memory_agent, ms=ms)
    agent.update_state()
