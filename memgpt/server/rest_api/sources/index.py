import uuid
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Query, HTTPException, status, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import SourceModel, PassageModel, DocumentModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.data_types import Source
from memgpt.data_sources.connectors import DirectoryConnector

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
    description: Optional[str] = Field(None, description="The description of the source.")


class CreateSourceResponse(BaseModel):
    source: SourceModel = Field(..., description="The newly created source.")


class UploadFileToSourceRequest(BaseModel):
    file: UploadFile = Field(..., description="The file to upload.")


class UploadFileToSourceResponse(BaseModel):
    source: SourceModel = Field(..., description="The source the file was uploaded to.")
    added_passages: int = Field(..., description="The number of passages added to the source.")
    added_documents: int = Field(..., description="The number of documents added to the source.")


class GetSourcePassagesResponse(BaseModel):
    passages: List[PassageModel] = Field(..., description="List of passages from the source.")


class GetSourceDocumentsResponse(BaseModel):
    documents: List[DocumentModel] = Field(..., description="List of documents from the source.")


def setup_sources_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/sources", tags=["sources"], response_model=ListSourcesResponse)
    async def list_source(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        # Clear the interface
        interface.clear()

        sources = server.ms.list_sources(user_id=user_id)
        return ListSourcesResponse(sources=sources)

    @router.post("/sources", tags=["sources"], response_model=SourceModel)
    async def create_source(
        request: CreateSourceRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        # TODO: don't use Source and just use SourceModel once pydantic migration is complete
        source = server.create_source(name=request.name, user_id=user_id)
        return SourceModel(
            name=source.name,
            description=None,  # TODO: actually store descriptions
            user_id=source.user_id,
            id=source.id,
            embedding_config=server.server_embedding_config,
            created_at=source.created_at.timestamp(),
        )

    @router.delete("/sources/{source_id}", tags=["sources"])
    async def delete_source(
        source_id,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        try:
            server.delete_source(source_id=uuid.UUID(source_id), user_id=user_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Source source_id={source_id} successfully deleted"})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.post("/sources/attach", tags=["sources"], response_model=SourceModel)
    async def attach_source_to_agent(
        agent_id: uuid.UUID = Query(..., description="The unique identifier of the agent to attach the source to."),
        source_name: str = Query(..., description="The name of the source to attach."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        assert isinstance(agent_id, uuid.UUID), f"Expected agent_id to be a UUID, got {agent_id}"
        assert isinstance(user_id, uuid.UUID), f"Expected user_id to be a UUID, got {user_id}"
        source = server.attach_source_to_agent(source_name=source_name, agent_id=agent_id, user_id=user_id)
        return SourceModel(
            name=source.name,
            description=None,  # TODO: actually store descriptions
            user_id=source.user_id,
            id=source.id,
            embedding_config=server.server_embedding_config,
            created_at=source.created_at,
        )

    @router.post("/sources/detach", tags=["sources"], response_model=SourceModel)
    async def detach_source_from_agent(
        agent_id: uuid.UUID = Query(..., description="The unique identifier of the agent to detach the source from."),
        source_name: str = Query(..., description="The name of the source to detach."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        server.detach_source_from_agent(source_name=source_name, agent_id=agent_id, user_id=user_id)

    @router.post("/sources/upload", tags=["sources"], response_model=UploadFileToSourceResponse)
    async def upload_file_to_source(
        # file: UploadFile = UploadFile(..., description="The file to upload."),
        file: UploadFile,
        source_id: uuid.UUID = Query(..., description="The unique identifier of the source to attach."),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        interface.clear()
        source = server.ms.get_source(source_id=source_id, user_id=user_id)

        # create a directory connector that reads the in-memory  file
        connector = DirectoryConnector(input_files=[file.filename])

        # load the data into the source via the connector
        server.load_data(user_id=user_id, source_name=source.name, connector=connector)

        # TODO: actually return added passages/documents
        return UploadFileToSourceResponse(source=source, added_passages=0, added_documents=0)

    @router.get("/sources/passages ", tags=["sources"], response_model=GetSourcePassagesResponse)
    async def list_passages(
        source_id: uuid.UUID = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        raise NotImplementedError

    @router.get("/sources/documents", tags=["sources"], response_model=GetSourceDocumentsResponse)
    async def list_documents(
        source_id: uuid.UUID = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        raise NotImplementedError

    return router
