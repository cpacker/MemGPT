from abc import ABC, abstractmethod
from typing import List

from memgpt.memory import (
    BaseRecallMemory,
    EmbeddingArchivalMemory,
)
from memgpt.utils import get_local_time, printd
from memgpt.data_types import Message, ToolCall, AgentState

from datetime import datetime


def parse_formatted_time(formatted_time: str):
    # parse times returned by memgpt.utils.get_formatted_time()
    try:
        return datetime.strptime(formatted_time.strip(), "%Y-%m-%d %I:%M:%S %p %Z%z")
    except:
        return datetime.strptime(formatted_time.strip(), "%Y-%m-%d %I:%M:%S %p")


class PersistenceManager(ABC):
    @abstractmethod
    def trim_messages(self, num):
        pass

    @abstractmethod
    def prepend_to_messages(self, added_messages):
        pass

    @abstractmethod
    def append_to_messages(self, added_messages):
        pass

    @abstractmethod
    def swap_system_message(self, new_system_message):
        pass

    @abstractmethod
    def update_memory(self, new_memory):
        pass


class LocalStateManager(PersistenceManager):
    """In-memory state manager has nothing to manage, all agents are held in-memory"""

    recall_memory_cls = BaseRecallMemory
    archival_memory_cls = EmbeddingArchivalMemory

    def __init__(self, agent_state: AgentState):
        # Memory held in-state useful for debugging stateful versions
        self.memory = None
        # self.messages = []  # current in-context messages
        # self.all_messages = [] # all messages seen in current session (needed if lazily synchronizing state with DB)
        self.archival_memory = EmbeddingArchivalMemory(agent_state)
        self.recall_memory = BaseRecallMemory(agent_state)
        # self.agent_state = agent_state

    def save(self):
        """Ensure storage connectors save data"""
        self.archival_memory.save()
        self.recall_memory.save()

    def init(self, agent):
        """Connect persistent state manager to agent"""
        printd(f"Initializing {self.__class__.__name__} with agent object")
        # self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        # self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        # printd(f"{self.__class__.__name__}.all_messages.len = {len(self.all_messages)}")
        printd(f"{self.__class__.__name__}.messages.len = {len(self.messages)}")

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

    def prepend_to_messages(self, added_messages: List[Message]):
        # first tag with timestamps
        # added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.prepend_to_message")
        # self.messages = [self.messages[0]] + added_messages + self.messages[1:]

        # add to recall memory
        self.recall_memory.insert_many([m for m in added_messages])

    def append_to_messages(self, added_messages: List[Message]):
        # first tag with timestamps
        # added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.append_to_messages")
        # self.messages = self.messages + added_messages

        # add to recall memory
        self.recall_memory.insert_many([m for m in added_messages])

    def swap_system_message(self, new_system_message: Message):
        # first tag with timestamps
        # new_system_message = {"timestamp": get_local_time(), "message": new_system_message}

        printd(f"{self.__class__.__name__}.swap_system_message")
        # self.messages[0] = new_system_message

        # add to recall memory
        self.recall_memory.insert(new_system_message)

    def update_memory(self, new_memory):
        printd(f"{self.__class__.__name__}.update_memory")
        self.memory = new_memory
