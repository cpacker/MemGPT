import uuid
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class CommandRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier of the user issuing the command.")
    agent_id: str = Field(..., description="Identifier of the agent on which the command will be executed.")
    command: str = Field(..., description="The command to be executed by the agent.")


class CommandResponse(BaseModel):
    response: str = Field(..., description="The result of the executed command.")


def setup_agents_command_router(server: SyncServer, interface: QueuingInterface):
    @router.post("/agents/command", tags=["agents"], response_model=CommandResponse)
    def run_command(request: CommandRequest = Body(...)):
        """
        Execute a command on a specified agent.

        This endpoint receives a command to be executed on an agent. It uses the user and agent identifiers to authenticate and route the command appropriately.

        Raises an HTTPException for any processing errors.
        """
        interface.clear()
        try:
            # TODO remove once chatui adds user selection / pulls user from config
            request.user_id = None if request.user_id == "null" else request.user_id

            user_id = uuid.UUID(request.user_id) if request.user_id else None
            agent_id = uuid.UUID(request.agent_id) if request.agent_id else None
            response = server.run_command(user_id=user_id, agent_id=agent_id, command=request.command)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return CommandResponse(response=response)

    return router
