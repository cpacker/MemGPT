from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

from letta.memory import BaseRecallMemory, EmbeddingArchivalMemory
from letta.schemas.agent import AgentState
from letta.schemas.memory import Memory
from letta.schemas.message import Message
from letta.utils import printd


def parse_formatted_time(formatted_time: str):
    # parse times returned by letta.utils.get_formatted_time()
    try:
        return datetime.strptime(formatted_time.strip(), "%Y-%m-%d %I:%M:%S %p %Z%z")
    except:
        return datetime.strptime(formatted_time.strip(), "%Y-%m-%d %I:%M:%S %p")


class PersistenceManager(ABC):
    @abstractmethod
    def trim_messages(self, num):
        pass

    @abstractmethod
    def update_memory(self, new_memory):
        pass


class LocalStateManager(PersistenceManager):
    """In-memory state manager has nothing to manage, all agents are held in-memory"""

    archival_memory_cls = EmbeddingArchivalMemory

    def __init__(self, agent_state: AgentState):
        # Memory held in-state useful for debugging stateful versions
        self.memory = agent_state.memory
        # self.messages = []  # current in-context messages
        # self.all_messages = [] # all messages seen in current session (needed if lazily synchronizing state with DB)
        self.archival_memory = EmbeddingArchivalMemory(agent_state)
        # self.agent_state = agent_state

    def save(self):
        """Ensure storage connectors save data"""
        self.archival_memory.save()

    '''
    def json_to_message(self, message_json) -> Message:
        """Convert agent message JSON into Message object"""

        # get message
        if "message" in message_json:
            message = message_json["message"]
        else:
            message = message_json

        # get timestamp
        if "timestamp" in message_json:
            timestamp = parse_formatted_time(message_json["timestamp"])
        else:
            timestamp = get_local_time()

        # TODO: change this when we fully migrate to tool calls API
        if "function_call" in message:
            tool_calls = [
                ToolCall(
                    id=message["tool_call_id"],
                    tool_call_type="function",
                    function={
                        "name": message["function_call"]["name"],
                        "arguments": message["function_call"]["arguments"],
                    },
                )
            ]
            printd(f"Saving tool calls {[vars(tc) for tc in tool_calls]}")
        else:
            tool_calls = None

        # if message["role"] == "function":
        # message["role"] = "tool"

        return Message(
            user_id=self.agent_state.user_id,
            agent_id=self.agent_state.id,
            role=message["role"],
            text=message["content"],
            name=message["name"] if "name" in message else None,
            model=self.agent_state.llm_config.model,
            created_at=timestamp,
            tool_calls=tool_calls,
            tool_call_id=message["tool_call_id"] if "tool_call_id" in message else None,
            id=message["id"] if "id" in message else None,
        )
    '''

    def trim_messages(self, num):
        # printd(f"InMemoryStateManager.trim_messages")
        # self.messages = [self.messages[0]] + self.messages[num:]
        pass

    def update_memory(self, new_memory: Memory):
        printd(f"{self.__class__.__name__}.update_memory")
        assert isinstance(new_memory, Memory), type(new_memory)
        self.memory = new_memory
