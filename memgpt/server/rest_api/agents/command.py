from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class Command(BaseModel):
    user_id: str
    agent_id: str
    command: str


def setup_agents_command_router(server: SyncServer, interface: QueuingInterface):
    @router.post("/agents/command")
    def run_command(body: Command):
        interface.clear()
        try:
            response = server.run_command(user_id=body.user_id, agent_id=body.agent_id, command=body.command)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return {"response": response}

    return router
