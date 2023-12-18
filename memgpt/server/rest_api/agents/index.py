from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class CreateAgentConfig(BaseModel):
    user_id: str
    config: dict


def setup_agents_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/agents", tags=["agents"])
    def list_agents(user_id: str):
        interface.clear()
        return server.list_agents(user_id=user_id)

    @router.post("/agents")
    def create_agent(body: CreateAgentConfig):
        interface.clear()
        try:
            agent_id = server.create_agent(user_id=body.user_id, agent_config=body.config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return {"agent_id": agent_id}

    return router
