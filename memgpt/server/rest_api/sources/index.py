import os
import tempfile
from functools import partial
from typing import List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    HTTPException,
    Query,
    UploadFile,
)

from memgpt.schemas.document import Document
from memgpt.schemas.job import Job
from memgpt.schemas.passage import Passage

# schemas
from memgpt.schemas.source import Source, SourceCreate, SourceUpdate, UploadFile
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

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


# class ListSourcesResponse(BaseModel):
#    sources: List[SourceModel] = Field(..., description="List of available sources.")
#
#
# class CreateSourceRequest(BaseModel):
#    name: str = Field(..., description="The name of the source.")
#    description: Optional[str] = Field(None, description="The description of the source.")
#
#
# class UploadFileToSourceRequest(BaseModel):
#    file: UploadFile = Field(..., description="The file to upload.")
#
#
# class UploadFileToSourceResponse(BaseModel):
#    source: SourceModel = Field(..., description="The source the file was uploaded to.")
#    added_passages: int = Field(..., description="The number of passages added to the source.")
#    added_documents: int = Field(..., description="The number of documents added to the source.")
#
#
# class GetSourcePassagesResponse(BaseModel):
#    passages: List[PassageModel] = Field(..., description="List of passages from the source.")
#
#
# class GetSourceDocumentsResponse(BaseModel):
#    documents: List[DocumentModel] = Field(..., description="List of documents from the source.")


def load_file_to_source_async(server: SyncServer, source_id: str, job_id: str, file: UploadFile, bytes: bytes):
    # write the file to a temporary directory (deleted after the context manager exits)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(bytes)

        server.load_file_to_source(source_id, file_path, job_id)


def setup_sources_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/sources/{source_id}", tags=["sources"], response_model=Source)
    async def get_source(
        source_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Get all sources
        """
        interface.clear()
        source = server.get_source(source_id=source_id, user_id=user_id)
        return source

    @router.get("/sources/name/{source_name}", tags=["sources"], response_model=str)
    async def get_source_id_by_name(
        source_name: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Get a source by name
        """
        interface.clear()
        source = server.get_source_id(source_name=source_name, user_id=user_id)
        return source

    @router.get("/sources", tags=["sources"], response_model=List[Source])
    async def list_sources(
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        List all data sources created by a user.
        """
        # Clear the interface
        interface.clear()

        try:
            sources = server.list_all_sources(user_id=user_id)
            return sources
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.post("/sources", tags=["sources"], response_model=Source)
    async def create_source(
        request: SourceCreate = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Create a new data source.
        """
        interface.clear()
        try:
            return server.create_source(request=request, user_id=user_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.post("/sources/{source_id}", tags=["sources"], response_model=Source)
    async def update_source(
        source_id: str,
        request: SourceUpdate = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Update the name or documentation of an existing data source.
        """
        interface.clear()
        try:
            return server.update_source(request=request, user_id=user_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.delete("/sources/{source_id}", tags=["sources"])
    async def delete_source(
        source_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Delete a data source.
        """
        interface.clear()
        try:
            server.delete_source(source_id=source_id, user_id=user_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.post("/sources/{source_id}/attach", tags=["sources"], response_model=Source)
    async def attach_source_to_agent(
        source_id: str,
        agent_id: str = Query(..., description="The unique identifier of the agent to attach the source to."),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Attach a data source to an existing agent.
        """
        interface.clear()
        assert isinstance(agent_id, str), f"Expected agent_id to be a UUID, got {agent_id}"
        assert isinstance(user_id, str), f"Expected user_id to be a UUID, got {user_id}"
        source = server.ms.get_source(source_id=source_id, user_id=user_id)
        source = server.attach_source_to_agent(source_id=source.id, agent_id=agent_id, user_id=user_id)
        return source

    @router.post("/sources/{source_id}/detach", tags=["sources"])
    async def detach_source_from_agent(
        source_id: str,
        agent_id: str = Query(..., description="The unique identifier of the agent to detach the source from."),
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Detach a data source from an existing agent.
        """
        server.detach_source_from_agent(source_id=source_id, agent_id=agent_id, user_id=user_id)

    @router.get("/sources/status/{job_id}", tags=["sources"], response_model=Job)
    async def get_job(
        job_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Get the status of a job.
        """
        job = server.get_job(job_id=job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job with id={job_id} not found.")
        return job

    @router.post("/sources/{source_id}/upload", tags=["sources"], response_model=Job)
    async def upload_file_to_source(
        # file: UploadFile = UploadFile(..., description="The file to upload."),
        file: UploadFile,
        source_id: str,
        background_tasks: BackgroundTasks,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        Upload a file to a data source.
        """
        interface.clear()
        source = server.ms.get_source(source_id=source_id, user_id=user_id)
        bytes = file.file.read()

        # create job
        # TODO: create server function
        job = Job(user_id=user_id, metadata_={"type": "embedding", "filename": file.filename, "source_id": source_id})
        job_id = job.id
        server.ms.create_job(job)

        # create background task
        background_tasks.add_task(load_file_to_source_async, server, source_id=source.id, job_id=job.id, file=file, bytes=bytes)

        # return job information
        job = server.ms.get_job(job_id=job_id)
        return job

    @router.get("/sources/{source_id}/passages ", tags=["sources"], response_model=List[Passage])
    async def list_passages(
        source_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        List all passages associated with a data source.
        """
        # TODO: check if paginated?
        passages = server.list_data_source_passages(user_id=user_id, source_id=source_id)
        return passages

    @router.get("/sources/{source_id}/documents", tags=["sources"], response_model=List[Document])
    async def list_documents(
        source_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        """
        List all documents associated with a data source.
        """
        documents = server.list_data_source_documents(user_id=user_id, source_id=source_id)
        return documents

    return router
