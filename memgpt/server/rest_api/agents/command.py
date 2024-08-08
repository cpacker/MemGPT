from functools import partial
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from memgpt.schemas.message import Message
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


class CommandRequest(BaseModel):
    command: str = Field(..., description="The command to be executed by the agent.")


class CommandResponse(BaseModel):
    response: str = Field(..., description="The result of the executed command.")


def setup_agents_command_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.post("/agents/{agent_id}/command", tags=["agents"], response_model=List[Message])
    def run_command(
        agent_id: str,
        request: Message = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Execute a command on a specified agent.

        This endpoint receives a command to be executed on an agent. It uses the user and agent identifiers to authenticate and route the command appropriately.

        Raises an HTTPException for any processing errors.
        """
        interface.clear()
        try:
            response = server.run_command(user_id=user_id, agent_id=agent_id, command=request.command)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return CommandResponse(response=response)

    return router
