import os
import tempfile
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, Query, UploadFile

from memgpt.schemas.document import Document
from memgpt.schemas.job import Job
from memgpt.schemas.passage import Passage
from memgpt.schemas.source import Source, SourceCreate, SourceUpdate
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.server import SyncServer

# These can be forward refs, but because Fastapi needs them at runtime the must be imported normally


router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/{source_id}", response_model=Source, operation_id="get_source")
def get_source(
    source_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get all sources
    """
    actor = server.get_current_user()

    return server.get_source(source_id=source_id, user_id=actor.id)


@router.get("/name/{source_name}", response_model=str, operation_id="get_source_id_by_name")
def get_source_id_by_name(
    source_name: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get a source by name
    """
    actor = server.get_current_user()

    source_id = server.get_source_id(source_name=source_name, user_id=actor.id)
    return source_id


@router.get("/", response_model=List[Source], operation_id="list_sources")
def list_sources(
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all data sources created by a user.
    """
    actor = server.get_current_user()

    return server.list_all_sources(user_id=actor.id)


@router.post("/", response_model=Source, operation_id="create_source")
def create_source(
    source: SourceCreate,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Create a new data source.
    """
    actor = server.get_current_user()

    return server.create_source(request=source, user_id=actor.id)


@router.patch("/{source_id}", response_model=Source, operation_id="update_source")
def update_source(
    source_id: str,
    source: SourceUpdate,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Update the name or documentation of an existing data source.
    """
    actor = server.get_current_user()
    assert source.id == source_id, "Source ID in path must match ID in request body"

    return server.update_source(request=source, user_id=actor.id)


@router.delete("/{source_id}", response_model=None, operation_id="delete_source")
def delete_source(
    source_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Delete a data source.
    """
    actor = server.get_current_user()

    server.delete_source(source_id=source_id, user_id=actor.id)


@router.post("/{source_id}/attach", response_model=Source, operation_id="attach_agent_to_source")
def attach_source_to_agent(
    source_id: str,
    agent_id: str = Query(..., description="The unique identifier of the agent to attach the source to."),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Attach a data source to an existing agent.
    """
    actor = server.get_current_user()

    source = server.ms.get_source(source_id=source_id, user_id=actor.id)
    assert source is not None, f"Source with id={source_id} not found."
    source = server.attach_source_to_agent(source_id=source.id, agent_id=agent_id, user_id=actor.id)
    return source


@router.post("/{source_id}/detach", response_model=None, operation_id="detach_agent_from_source")
def detach_source_from_agent(
    source_id: str,
    agent_id: str = Query(..., description="The unique identifier of the agent to detach the source from."),
    server: "SyncServer" = Depends(get_memgpt_server),
) -> None:
    """
    Detach a data source from an existing agent.
    """
    actor = server.get_current_user()

    server.detach_source_from_agent(source_id=source_id, agent_id=agent_id, user_id=actor.id)


@router.post("/{source_id}/upload", response_model=Job, operation_id="upload_file_to_source")
def upload_file_to_source(
    file: UploadFile,
    source_id: str,
    background_tasks: BackgroundTasks,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Upload a file to a data source.
    """
    actor = server.get_current_user()

    source = server.ms.get_source(source_id=source_id, user_id=actor.id)
    assert source is not None, f"Source with id={source_id} not found."
    bytes = file.file.read()

    # create job
    job = Job(
        user_id=actor.id,
        metadata_={"type": "embedding", "filename": file.filename, "source_id": source_id},
        completed_at=None,
    )
    job_id = job.id
    server.ms.create_job(job)

    # create background task
    background_tasks.add_task(load_file_to_source_async, server, source_id=source.id, job_id=job.id, file=file, bytes=bytes)

    # return job information
    job = server.ms.get_job(job_id=job_id)
    assert job is not None, "Job not found"
    return job


@router.get("/{source_id}/passages", response_model=List[Passage], operation_id="list_source_passages")
def list_passages(
    source_id: str,
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    List all passages associated with a data source.
    """
    actor = server.get_current_user()
    passages = server.list_data_source_passages(user_id=actor.id, source_id=source_id)
    return passages


@router.get("/{source_id}/documents", response_model=List[Document], operation_id="list_source_documents")
def list_documents(
    source_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all documents associated with a data source.
    """
    actor = server.get_current_user()

    documents = server.list_data_source_documents(user_id=actor.id, source_id=source_id)
    return documents


def load_file_to_source_async(server: SyncServer, source_id: str, job_id: str, file: UploadFile, bytes: bytes):
    # write the file to a temporary directory (deleted after the context manager exits)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(str(tmpdirname), str(file.filename))
        with open(file_path, "wb") as buffer:
            buffer.write(bytes)

        server.load_file_to_source(source_id, file_path, job_id)
