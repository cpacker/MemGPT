from fastapi import APIRouter
from pydantic import BaseModel

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class CoreMemory(BaseModel):
    user_id: str
    agent_id: str
    human: str | None = None
    persona: str | None = None


def setup_agents_memory_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents/memory")
    def get_agent_memory(user_id: str, agent_id: str):
        interface.clear()
        return server.get_agent_memory(user_id=user_id, agent_id=agent_id)

    @router.put("/agents/memory")
    def get_agent_memory(body: CoreMemory):
        interface.clear()
        new_memory_contents = {"persona": body.persona, "human": body.human}
        return server.update_agent_core_memory(user_id=body.user_id, agent_id=body.agent_id, new_memory_contents=new_memory_contents)

    return router
