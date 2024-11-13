from typing import List, Optional, Union

from letta.agent import Agent
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.schemas.agent import AgentState
from letta.schemas.message import Message
from letta.schemas.tool import Tool
from letta.schemas.usage import LettaUsageStatistics


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
        """Go through what is currently in memory core memory and information."""
