from abc import ABC, abstractmethod
import pickle
from memgpt.config import AgentConfig
from memgpt.memory import (
    DummyRecallMemory,
    EmbeddingArchivalMemory,
)
from memgpt.utils import get_local_time, printd


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

    recall_memory_cls = DummyRecallMemory
    archival_memory_cls = EmbeddingArchivalMemory

    def __init__(self, agent_config: AgentConfig):
        # Memory held in-state useful for debugging stateful versions
        self.memory = None
        self.messages = []
        self.all_messages = []
        self.archival_memory = EmbeddingArchivalMemory(agent_config)
        self.recall_memory = None
        self.agent_config = agent_config

    @classmethod
    def load(cls, filename, agent_config: AgentConfig):
        """ Load a LocalStateManager from a file. """ ""
        with open(filename, "rb") as f:
            data = pickle.load(f)

        manager = cls(agent_config)
        manager.all_messages = data["all_messages"]
        manager.messages = data["messages"]
        manager.recall_memory = data["recall_memory"]
        manager.archival_memory = EmbeddingArchivalMemory(agent_config)
        return manager

    def save(self, filename):
        with open(filename, "wb") as fh:
            ## TODO: fix this hacky solution to pickle the retriever
            self.archival_memory.save()
            pickle.dump(
                {
                    "recall_memory": self.recall_memory,
                    "messages": self.messages,
                    "all_messages": self.all_messages,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            printd(f"Saved state to {fh}")

    def init(self, agent):
        printd(f"Initializing {self.__class__.__name__} with agent object")
        self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        printd(f"{self.__class__.__name__}.all_messages.len = {len(self.all_messages)}")
        printd(f"{self.__class__.__name__}.messages.len = {len(self.messages)}")

        # Persistence manager also handles DB-related state
        self.recall_memory = self.recall_memory_cls(message_database=self.all_messages)

        # TODO: init archival memory here?

    def trim_messages(self, num):
        # printd(f"InMemoryStateManager.trim_messages")
        self.messages = [self.messages[0]] + self.messages[num:]

    def prepend_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.prepend_to_message")
        self.messages = [self.messages[0]] + added_messages + self.messages[1:]
        self.all_messages.extend(added_messages)

    def append_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"{self.__class__.__name__}.append_to_messages")
        self.messages = self.messages + added_messages
        self.all_messages.extend(added_messages)

    def swap_system_message(self, new_system_message):
        # first tag with timestamps
        new_system_message = {"timestamp": get_local_time(), "message": new_system_message}

        printd(f"{self.__class__.__name__}.swap_system_message")
        self.messages[0] = new_system_message
        self.all_messages.append(new_system_message)

    def update_memory(self, new_memory):
        printd(f"{self.__class__.__name__}.update_memory")
        self.memory = new_memory
