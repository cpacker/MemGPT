import uuid
from functools import partial
from typing import List

from fastapi import APIRouter, Depends, Body, UploadFile
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import SourceModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.data_types import Source

router = APIRouter()

"""
Implement the following functions:
* List all available sources
* Create a new source
* Delete a source
* Upload a file to a server that is loaded into a specific source
* Paginated get all passages from a source
* Paginated get all documents from a source
* Attach a source to an agent
"""


class ListSourcesResponse(BaseModel):
    sources: List[SourceModel] = Field(..., description="List of available sources")


class CreateSourceRequest(BaseModel):
    name: str = Field(..., description="The name of the source.")
    description: str = Field(..., description="The description of the source.")


class CreateSourceResponse(BaseModel):
    source: SourceModel = Field(..., description="The newly created source.")


class DeleteSourceRequest(BaseModel):
    source_id: uuid.UUID = Field(..., description="The unique identifier of the source to delete.")


class DeleteSourceResponse(BaseModel):
    source: SourceModel = Field(..., description="The deleted source.")


class AttachSourceToAgentRequest(BaseModel):
    source_id: uuid.UUID = Field(..., description="The unique identifier of the source to attach.")
    agent_id: uuid.UUID = Field(..., description="The unique identifier of the agent to attach the source to.")


class AttachSourceToAgentResponse(BaseModel):
    source: SourceModel = Field(..., description="The attached source.")


class UploadFileToSourceRequest(BaseModel):
    source_id: uuid.UUID = Field(..., description="The unique identifier of the source to attach.")
    file: UploadFile = Field(..., description="The file to upload.")


class UploadFileToSourceResponse(BaseModel):
    source: SourceModel = Field(..., description="The source the file was uploaded to.")
    added_passages: int = Field(..., description="The number of passages added to the source.")
    added_documents: int = Field(..., description="The number of documents added to the source.")


def setup_personas_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/sources", tags=["sources"], response_model=ListSourcesResponse)
    async def list_personas(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()

        sources = server.ms.list_sources(user_id=user_id)
        return ListSourcesResponse(sources=sources)

    @router.post("/sources ", tags=["sources"], response_model=SourceModel)
    async def create_persona(
        request: CreateSourceRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        # TODO: don't use Source and just use SourceModel once pydantic migration is complete
        source = Source(
            name=request.name,
            user_id=user_id,
        )
        server.ms.add_source(source)
        return SourceModel(
            name=source.name,
            description=None,  # TODO: actually store descriptions
            user_id=source.user_id,
            id=source.id,
            embedding_config=server.server_embedding_config,
            created_at=source.created_at,
        )

    return router
