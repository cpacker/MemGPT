import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Depends

from memgpt.models.pydantic_models import SourceModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_agents_sources_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/agents/{agent_id}/sources", tags=["agents"], response_model=List[SourceModel])
    def get_agent_sources(
        agent_id: uuid.UUID,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """
        Retrieve all the tools associated with an agent.
        """
        agent_state = server.ms.get_agent(agent_id=agent_id, user_id=user_id)
        sources_ids = server.ms.list_attached_sources(agent_id=agent_state.id)
        sources = [server.ms.get_source(source_id=s_id) for s_id in sources_ids]
        return sources
