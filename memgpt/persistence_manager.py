from abc import ABC, abstractmethod
import os
import pickle
from memgpt.config import AgentConfig
from .memory import (
    DummyRecallMemory,
    DummyRecallMemoryWithEmbeddings,
    DummyArchivalMemory,
    DummyArchivalMemoryWithEmbeddings,
    DummyArchivalMemoryWithFaiss,
    EmbeddingArchivalMemory,
)
from .utils import get_local_time, printd


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


class InMemoryStateManager(PersistenceManager):
    """In-memory state manager has nothing to manage, all agents are held in-memory"""

    recall_memory_cls = DummyRecallMemory
    archival_memory_cls = DummyArchivalMemory

    def __init__(self):
        # Memory held in-state useful for debugging stateful versions
        self.memory = None
        self.messages = []
        self.all_messages = []

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def save(self, filename):
        with open(filename, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def init(self, agent):
        printd(f"Initializing InMemoryStateManager with agent object")
        self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        printd(f"InMemoryStateManager.all_messages.len = {len(self.all_messages)}")
        printd(f"InMemoryStateManager.messages.len = {len(self.messages)}")

        # Persistence manager also handles DB-related state
        self.recall_memory = self.recall_memory_cls(message_database=self.all_messages)
        self.archival_memory_db = []
        self.archival_memory = self.archival_memory_cls(archival_memory_database=self.archival_memory_db)

    def trim_messages(self, num):
        # printd(f"InMemoryStateManager.trim_messages")
        self.messages = [self.messages[0]] + self.messages[num:]

    def prepend_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"InMemoryStateManager.prepend_to_message")
        self.messages = [self.messages[0]] + added_messages + self.messages[1:]
        self.all_messages.extend(added_messages)

    def append_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"InMemoryStateManager.append_to_messages")
        self.messages = self.messages + added_messages
        self.all_messages.extend(added_messages)

    def swap_system_message(self, new_system_message):
        # first tag with timestamps
        new_system_message = {"timestamp": get_local_time(), "message": new_system_message}

        printd(f"InMemoryStateManager.swap_system_message")
        self.messages[0] = new_system_message
        self.all_messages.append(new_system_message)

    def update_memory(self, new_memory):
        printd(f"InMemoryStateManager.update_memory")
        self.memory = new_memory


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
        printd(f"Initializing InMemoryStateManager with agent object")
        self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        printd(f"InMemoryStateManager.all_messages.len = {len(self.all_messages)}")
        printd(f"InMemoryStateManager.messages.len = {len(self.messages)}")

        # Persistence manager also handles DB-related state
        self.recall_memory = self.recall_memory_cls(message_database=self.all_messages)

        # TODO: init archival memory here?

    def trim_messages(self, num):
        # printd(f"InMemoryStateManager.trim_messages")
        self.messages = [self.messages[0]] + self.messages[num:]

    def prepend_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"InMemoryStateManager.prepend_to_message")
        self.messages = [self.messages[0]] + added_messages + self.messages[1:]
        self.all_messages.extend(added_messages)

    def append_to_messages(self, added_messages):
        # first tag with timestamps
        added_messages = [{"timestamp": get_local_time(), "message": msg} for msg in added_messages]

        printd(f"InMemoryStateManager.append_to_messages")
        self.messages = self.messages + added_messages
        self.all_messages.extend(added_messages)

    def swap_system_message(self, new_system_message):
        # first tag with timestamps
        new_system_message = {"timestamp": get_local_time(), "message": new_system_message}

        printd(f"InMemoryStateManager.swap_system_message")
        self.messages[0] = new_system_message
        self.all_messages.append(new_system_message)

    def update_memory(self, new_memory):
        printd(f"InMemoryStateManager.update_memory")
        self.memory = new_memory


class InMemoryStateManagerWithPreloadedArchivalMemory(InMemoryStateManager):
    archival_memory_cls = DummyArchivalMemory
    recall_memory_cls = DummyRecallMemory

    def __init__(self, archival_memory_db):
        self.archival_memory_db = archival_memory_db

    def init(self, agent):
        print(f"Initializing InMemoryStateManager with agent object")
        self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        print(f"InMemoryStateManager.all_messages.len = {len(self.all_messages)}")
        print(f"InMemoryStateManager.messages.len = {len(self.messages)}")
        self.recall_memory = self.recall_memory_cls(message_database=self.all_messages)
        self.archival_memory = self.archival_memory_cls(archival_memory_database=self.archival_memory_db)


class InMemoryStateManagerWithEmbeddings(InMemoryStateManager):
    archival_memory_cls = DummyArchivalMemoryWithEmbeddings
    recall_memory_cls = DummyRecallMemoryWithEmbeddings


class InMemoryStateManagerWithFaiss(InMemoryStateManager):
    archival_memory_cls = DummyArchivalMemoryWithFaiss
    recall_memory_cls = DummyRecallMemoryWithEmbeddings

    def __init__(self, archival_index, archival_memory_db, a_k=100):
        super().__init__()
        self.archival_index = archival_index
        self.archival_memory_db = archival_memory_db
        self.a_k = a_k

    def save(self, _filename):
        raise NotImplementedError

    def init(self, agent):
        print(f"Initializing InMemoryStateManager with agent object")
        self.all_messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.messages = [{"timestamp": get_local_time(), "message": msg} for msg in agent.messages.copy()]
        self.memory = agent.memory
        print(f"InMemoryStateManager.all_messages.len = {len(self.all_messages)}")
        print(f"InMemoryStateManager.messages.len = {len(self.messages)}")

        # Persistence manager also handles DB-related state
        self.recall_memory = self.recall_memory_cls(message_database=self.all_messages)
        self.archival_memory = self.archival_memory_cls(
            index=self.archival_index, archival_memory_database=self.archival_memory_db, k=self.a_k
        )
