from typing import List, Optional, Union

from letta.agent import Agent, save_agent
from letta.interface import AgentInterface
from letta.schemas.agent import AgentState
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import UsageStatistics
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User


def send_thinking_message(self: "Agent", message: str) -> Optional[str]:
    """
    Sends a thinking message so that the model can reason out loud before responding.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.internal_monologue(message)
    return None


def send_final_message(self: "Agent", message: str) -> Optional[str]:
    """
    Sends a final message to the human user after thinking for a while.

    Args:
        message (str): Message contents. All unicode (including emojis) are supported.

    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """
    self.interface.internal_monologue(message)
    return None


class O1Agent(Agent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        user: User,
        max_thinking_steps: int = 10,
        first_message_verify_mono: bool = False,
    ):
        super().__init__(interface, agent_state, user)
        self.max_thinking_steps = max_thinking_steps
        self.first_message_verify_mono = first_message_verify_mono

    def step(
        self,
        messages: Union[Message, List[Message]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        **kwargs,
    ) -> LettaUsageStatistics:
        """Run Agent.inner_step in a loop, terminate when final thinking message is sent or max_thinking_steps is reached"""
        # assert ms is not None, "MetadataStore is required"
        next_input_message = messages if isinstance(messages, list) else [messages]

        counter = 0
        total_usage = UsageStatistics()
        step_count = 0
        while step_count < self.max_thinking_steps:
            if counter > 0:
                next_input_message = []

            kwargs["first_message"] = False
            step_response = self.inner_step(
                messages=next_input_message,
                **kwargs,
            )
            usage = step_response.usage
            step_count += 1
            total_usage += usage
            counter += 1
            self.interface.step_complete()
            # check if it is final thinking message
            if step_response.messages[-1].name == "send_final_message":
                break
            save_agent(self)

        return LettaUsageStatistics(**total_usage.model_dump(), step_count=step_count)
