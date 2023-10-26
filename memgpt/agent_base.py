from abc import ABC, abstractmethod


class AgentAsyncBase(ABC):
    @abstractmethod
    async def step(self, user_message):
        pass
