from abc import ABC, abstractmethod
from typing import List

from memgpt.memory import (
    BaseRecallMemory,
    EmbeddingArchivalMemory,
)
from memgpt.utils import printd
from memgpt.data_types import Message, AgentState


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


class LocalStateManager(PersistenceManager):
    """In-memory state manager has nothing to manage, all agents are held in-memory"""

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
