from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

from letta.agent import Agent

from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.prompts import gpt_system
from letta.schemas.agent import AgentState, AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import BasicBlockMemory, Block
from letta.schemas.message import Message
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
from letta.utils import get_persona_text


class ChatOnlyAgent(Agent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        user: User,
        first_message_verify_mono: bool = False,
        always_rethink_memory: bool = True,
        recent_convo_limit: int = 2000,
    ):
        super().__init__(interface, agent_state, user)
        self.first_message_verify_mono = first_message_verify_mono
        self.always_rethink_memory = always_rethink_memory
        self.offline_memory_agent = None
        self.recent_convo_limit = recent_convo_limit

    def step(
        self,
        messages: Union[Message, List[Message]],
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        ms: Optional[MetadataStore] = None,
        **kwargs,
    ) -> LettaUsageStatistics:
        # assert ms is not None, "MetadataStore is required"
        letta_statistics = super().step(messages=messages, chaining=chaining, max_chaining_steps=max_chaining_steps, ms=ms, **kwargs)

        if self.always_rethink_memory:

            def generate_offline_memory_agent():
                from letta.client.client import create_client

                client = create_client()
                if self.offline_memory_agent:
                    client.delete_agent(agent_id=self.offline_memory_agent.id)
                    self.offline_memory_agent = None

                conversation_human_block = self.agent_state.memory.get_block("chat_agent_human")
                conversation_persona_block = self.agent_state.memory.get_block("chat_agent_persona")
                offline_persona_block = Block(
                    name="offline_memory_persona",
                    label="offline_memory_persona",
                    value=get_persona_text("offline_memory_persona"),
                    limit=2000,
                )
                conversation_human_block_new = Block(
                    name="chat_agent_human_new", label="chat_agent_human_new", value=conversation_human_block.value, limit=2000
                )
                conversation_persona_block_new = Block(
                    name="chat_agent_persona_new", label="chat_agent_persona_new", value=conversation_persona_block.value, limit=2000
                )

                recent_convo = "".join([str(message) for message in self.messages[3:]])[-self.recent_convo_limit:]
                conversation_messages_block = Block(name="conversation_block", label="conversation_block", value=recent_convo, limit=self.recent_convo_limit)

                offline_memory = BasicBlockMemory(
                    blocks=[
                        offline_persona_block,
                        conversation_human_block,
                        conversation_persona_block,
                        conversation_human_block_new,
                        conversation_persona_block_new,
                        conversation_messages_block,
                    ]
                )

                self.offline_memory_agent = client.create_agent(
                    name="offline_memory_agent",
                    agent_type=AgentType.offline_memory_agent,
                    system=gpt_system.get_system_text("memgpt_offline_memory_chat"),
                    memory=offline_memory,
                    llm_config=LLMConfig.default_config("gpt-4"),
                    embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
                    tools=self.agent_state.metadata_.get("offline_memory_tools", []),
                    include_base_tools=False,
                )
                self.offline_memory_agent.memory.update_block_value(label="conversation_block", value=recent_convo)
                client.send_message(agent_id=self.offline_memory_agent.id, message="Reorganize the memory", role="user")
                client.delete_agent(agent_id=self.offline_memory_agent.id)
                self.offline_memory_agent = None

            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(generate_offline_memory_agent)

        return letta_statistics
