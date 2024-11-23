from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from letta.agent import Agent
from letta.interface import AgentInterface
from letta.metadata import MetadataStore
from letta.schemas.agent import AgentState
from letta.schemas.message import Message
from letta.schemas.openai.chat_completion_response import UsageStatistics
from letta.schemas.tool import Tool
from letta.schemas.usage import LettaUsageStatistics
from letta.schemas.user import User
from letta.schemas.agent import AgentType
from letta.utils import get_persona_text
from letta.schemas.memory import BasicBlockMemory, Block
from letta.offline_memory_agent import (
    finish_rethinking_memory,
    rethink_memory_convo,
)
from letta.constants import DEFAULT_HUMAN, DEFAULT_PERSONA

from letta.prompts import gpt_system
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.tool_rule import TerminalToolRule



class ChatOnlyAgent(Agent):
    def __init__(
        self,
        interface: AgentInterface,
        agent_state: AgentState,
        user: User,
        tools: List[Tool] = [],
        first_message_verify_mono: bool = False,
        always_rethink_memory: bool = True,
    ):
        super().__init__(interface, agent_state, tools, user)
        self.tools = tools
        self.first_message_verify_mono = first_message_verify_mono
        self.always_rethink_memory = always_rethink_memory
        self.offline_memory_agent = None

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
            def rethink_memory():
                from letta.client.client import create_client
                client = create_client()
                if self.offline_memory_agent:
                    client.delete_agent(agent_id=self.offline_memory_agent.id)
                    self.offline_memory_agent = None


                conversation_human_block = self.memory.get_block("chat_agent_human")
                conversation_persona_block = self.memory.get_block("chat_agent_persona")
                offline_persona_block = Block(
                    name="offline_memory_persona", label="offline_memory_persona", value=get_persona_text("offline_memory_persona"), limit=2000
                )
                conversation_human_block_new = Block(
                    name="chat_agent_human_new", label="chat_agent_human_new", value=conversation_human_block.value, limit=2000
                )
                conversation_persona_block_new = Block(
                    name="chat_agent_persona_new", label="chat_agent_persona_new", value=conversation_persona_block.value, limit=2000
                )
                conversation_messages_block = Block(name="conversation_block", label="conversation_block", value="", limit=2000)

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

                rethink_memory_tool = client.create_tool(rethink_memory_convo)
                finish_rethinking_memory_tool = client.create_tool(finish_rethinking_memory)
                self.offline_memory_agent = client.create_agent(
                    name="offline_memory_agent",
                    agent_type=AgentType.offline_memory_agent,
                    system=gpt_system.get_system_text("memgpt_offline_memory_chat"),
                    memory=offline_memory,
                    llm_config=LLMConfig.default_config("gpt-4"),
                    embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
                    tools=[rethink_memory_tool.name, finish_rethinking_memory_tool.name],
                    tool_rules=[TerminalToolRule(tool_name=finish_rethinking_memory_tool.name)],
                    include_base_tools=False,
                )

                conversation_block_limit = 2000 
                recent_convo = "".join([str(message) for message in self.messages[3:]])[-conversation_block_limit:]
                self.offline_memory_agent.memory.update_block_value(label="conversation_block", value=recent_convo)
                block_id = self.offline_memory_agent.memory.get_block("conversation_block").id
                client.update_block(block_id, text=recent_convo)
                client.update_agent(agent_id=self.offline_memory_agent.id, memory=self.offline_memory_agent.memory)
                client.get_agent(self.offline_memory_agent.id)
                client.send_message(agent_id=self.offline_memory_agent.id, message="Reorganize the memory", role="user")
                client.delete_agent(agent_id=self.offline_memory_agent.id)
                self.update_memory_blocks_from_db()
                # client.get_agent(self.agent_state.id)
                # client.update_agent(agent_id=self.agent_state.id, memory=self.agent_state.memory)
                self.offline_memory_agent = None

            # Run the rethink_memory function in a separate thread
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(rethink_memory)

        return letta_statistics