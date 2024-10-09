import datetime
from typing import List, Optional, Union

from letta.agent import Agent
from letta.constants import FIRST_MESSAGE_ATTEMPTS
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.schemas.agent import AgentState, AgentStepResponse
from letta.schemas.enums import OptionState
from letta.schemas.message import Message
from letta.schemas.tool import Tool


def send_thinking_message(self: Agent, message: str) -> Optional[str]:
    """
    Sends a thinking message so that the model can reason out loud before responding.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    return None


def send_final_message(self: Agent, message: str) -> Optional[str]:
    """
    Sends a final message to the human user after thinking for a while.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.assistant_message(message)  # , msg_obj=self._messages[-1])
    return None


class O1Agent(Agent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        tools: List[Tool] = [],
        max_thinking_steps: int = 5,
        first_message_verify_mono: bool = False,
    ):
        super().__init__(interface, agent_state, tools)
        self.max_thinking_steps = max_thinking_steps
        # self.interface = interface
        # self.agent_state = agent_state
        self.tools = tools
        self.first_message_verify_mono = first_message_verify_mono

    def step(
        self,
        user_message: Union[Message, None, str],  # NOTE: should be json.dump(dict)
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
        stream: bool = False,  # TODO move to config?
        timestamp: Optional[datetime.datetime] = None,
        inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
        ms: Optional[MetadataStore] = None,
    ) -> AgentStepResponse:

        thinking_agent = Agent(
            interface=self.interface,
            agent_state=self.agent_state,
            tools=self.tools,
            first_message_verify_mono=self.first_message_verify_mono,
        )
        for _ in range(self.max_thinking_steps):
            # assert isinstance(self.agent_state.memory, Memory), f"Memory object is not of type Memory: {type(self.agent.agent_state.memory)}"

            # assert isinstance(thinking_agent.agent_state.memory, Memory), f"Memory object is not of type Memory"
            # print("THINKING AGENT", thinking_agent.agent_state.system)
            response = thinking_agent.step(
                user_message,
                first_message,
                first_message_retry_limit,
                skip_verify,
                return_dicts,
                recreate_message_timestamp,
                stream,
                timestamp,
                inner_thoughts_in_kwargs,
                ms,
            )
            if response.messages[-1].name == "send_final_message":
                break
        return response

    # def rebuild_memory(self, force=False, update_timestamp=True, ms: Optional[MetadataStore] = None):
    #    self.agent.rebuild_memory(force, update_timestamp, ms)

    # def update_state(self) -> AgentState:
    #    updated_state = self.agent.update_state()
    #    return updated_state
