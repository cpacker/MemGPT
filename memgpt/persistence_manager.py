from abc import ABC, abstractmethod
import pickle
from memgpt.config import AgentConfig
from memgpt.memory import (
    DummyRecallMemory,
    BaseRecallMemory,
    EmbeddingArchivalMemory,
)
from memgpt.utils import get_local_time, printd
from memgpt.data_types import Message
from memgpt.config import MemGPTConfig

from datetime import datetime


def parse_formatted_time(formatted_time):
    # parse times returned by memgpt.utils.get_formatted_time()
    return datetime.strptime(formatted_time, "%Y-%m-%d %I:%M:%S %p %Z%z")


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

    def __init__(self, agent_config: AgentConfig):
        # Memory held in-state useful for debugging stateful versions
        self.memory = None
        self.messages = []  # current in-context messages
        # self.all_messages = [] # all messages seen in current session (needed if lazily synchronizing state with DB)
        self.archival_memory = EmbeddingArchivalMemory(agent_config)
        self.recall_memory = BaseRecallMemory(agent_config)
        self.agent_config = agent_config
        self.config = MemGPTConfig.load()

    @classmethod
    def load(cls, agent_config: AgentConfig):
        """ Load a LocalStateManager from a file. """ ""
        # TODO: remove this class and just init the class
        manager = cls(agent_config)
        return manager

    def save(self):
        """Ensure storage connectors save data"""
        self.archival_memory.save()
        self.recall_memory.save()

    def init(self, agent):
        """Connect persistent state manager to agent"""
        printd(f"Initializing {self.__class__.__name__} with agent object")
        # self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        # printd(f"{self.__class__.__name__}.all_messages.len = {len(self.all_messages)}")
        printd(f"{self.__class__.__name__}.messages.len = {len(self.messages)}")

        # Persistence manager also handles DB-related state
        # self.recall_memory = self.recall_memory_cls(message_database=self.all_messages)

    def json_to_message(self, message_json) -> Message:
        """Convert agent message JSON into Message object"""
        timestamp = message_json["timestamp"]
        message = message_json["message"]

        return Message(
            user_id=self.config.anon_clientid,
            agent_id=self.agent_config.name,
            role=message["role"],
            text=message["content"],
            model=self.agent_config.model,
            created_at=parse_formatted_time(timestamp),
            function_name=message["function_name"] if "function_name" in message else None,
            function_args=message["function_args"] if "function_args" in message else None,
            function_response=message["function_response"] if "function_response" in message else None,
            id=message["id"] if "id" in message else None,
        )

    def trim_messages(self, num):
        # printd(f"InMemoryStateManager.trim_messages")
        self.messages = [self.messages[0]] + self.messages[num:]

    def prepend_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.prepend_to_message")
        self.messages = [self.messages[0]] + added_messages + self.messages[1:]

        # add to recall memory
        self.recall_memory.insert_many([self.json_to_message(m) for m in added_messages])

    def append_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.append_to_messages")
        self.messages = self.messages + added_messages

        # add to recall memory
        self.recall_memory.insert_many([self.json_to_message(m) for m in added_messages])

    def swap_system_message(self, new_system_message):
        # first tag with timestamps
        new_system_message = {"timestamp": get_local_time(), "message": new_system_message}

        printd(f"{self.__class__.__name__}.swap_system_message")
        self.messages[0] = new_system_message

        # add to recall memory
        self.recall_memory.insert(self.json_to_message(new_system_message))

    def update_memory(self, new_memory):
        printd(f"{self.__class__.__name__}.update_memory")
        self.memory = new_memory
