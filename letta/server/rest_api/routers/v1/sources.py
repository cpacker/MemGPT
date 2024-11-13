import os
import tempfile
from typing import List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Header,
    HTTPException,
    Query,
    UploadFile,
)

from letta.schemas.file import FileMetadata
from letta.schemas.job import Job
from letta.schemas.passage import Passage
from letta.schemas.source import Source, SourceCreate, SourceUpdate
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer

# These can be forward refs, but because Fastapi needs them at runtime the must be imported normally


router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/{source_id}", response_model=Source, operation_id="get_source")
def get_source(
    source_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Get all sources
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.source_manager.get_source_by_id(source_id=source_id, actor=actor)


@router.get("/name/{source_name}", response_model=str, operation_id="get_source_id_by_name")
def get_source_id_by_name(
    source_name: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Get a source by name
    """
    actor = server.get_user_or_default(user_id=user_id)

    source = server.source_manager.get_source_by_name(source_name=source_name, actor=actor)
    return source.id


@router.get("/", response_model=List[Source], operation_id="list_sources")
def list_sources(
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    List all data sources created by a user.
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.list_all_sources(actor=actor)


@router.post("/", response_model=Source, operation_id="create_source")
def create_source(
    source_create: SourceCreate,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Create a new data source.
    """
    actor = server.get_user_or_default(user_id=user_id)
    source = Source(**source_create.model_dump())

    return server.source_manager.create_source(source=source, actor=actor)


@router.patch("/{source_id}", response_model=Source, operation_id="update_source")
def update_source(
    source_id: str,
    source: SourceUpdate,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Update the name or documentation of an existing data source.
    """
    actor = server.get_user_or_default(user_id=user_id)
    return server.source_manager.update_source(source_id=source_id, source_update=source, actor=actor)


@router.delete("/{source_id}", response_model=None, operation_id="delete_source")
def delete_source(
    source_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Delete a data source.
    """
    actor = server.get_user_or_default(user_id=user_id)

    server.delete_source(source_id=source_id, actor=actor)


@router.post("/{source_id}/attach", response_model=Source, operation_id="attach_agent_to_source")
def attach_source_to_agent(
    source_id: str,
    agent_id: str = Query(..., description="The unique identifier of the agent to attach the source to."),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Attach a data source to an existing agent.
    """
    actor = server.get_user_or_default(user_id=user_id)

    source = server.source_manager.get_source_by_id(source_id=source_id, actor=actor)
    assert source is not None, f"Source with id={source_id} not found."
    source = server.attach_source_to_agent(source_id=source.id, agent_id=agent_id, user_id=actor.id)
    return source


@router.post("/{source_id}/detach", response_model=Source, operation_id="detach_agent_from_source")
def detach_source_from_agent(
    source_id: str,
    agent_id: str = Query(..., description="The unique identifier of the agent to detach the source from."),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
) -> None:
    """
    Detach a data source from an existing agent.
    """
    actor = server.get_user_or_default(user_id=user_id)

    return server.detach_source_from_agent(source_id=source_id, agent_id=agent_id, user_id=actor.id)


@router.post("/{source_id}/upload", response_model=Job, operation_id="upload_file_to_source")
def upload_file_to_source(
    file: UploadFile,
    source_id: str,
    background_tasks: BackgroundTasks,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Upload a file to a data source.
    """
    actor = server.get_user_or_default(user_id=user_id)

    source = server.source_manager.get_source_by_id(source_id=source_id, actor=actor)
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
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    List all passages associated with a data source.
    """
    actor = server.get_user_or_default(user_id=user_id)
    passages = server.list_data_source_passages(user_id=actor.id, source_id=source_id)
    return passages


@router.get("/{source_id}/files", response_model=List[FileMetadata], operation_id="list_files_from_source")
def list_files_from_source(
    source_id: str,
    limit: int = Query(1000, description="Number of files to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor to fetch the next set of results"),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    List paginated files associated with a data source.
    """
    actor = server.get_user_or_default(user_id=user_id)
    return server.source_manager.list_files(source_id=source_id, limit=limit, cursor=cursor, actor=actor)


# it's redundant to include /delete in the URL path. The HTTP verb DELETE already implies that action.
# it's still good practice to return a status indicating the success or failure of the deletion
@router.delete("/{source_id}/{file_id}", status_code=204, operation_id="delete_file_from_source")
def delete_file_from_source(
    source_id: str,
    file_id: str,
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Delete a data source.
    """
    actor = server.get_user_or_default(user_id=user_id)

    deleted_file = server.source_manager.delete_file(file_id=file_id, actor=actor)
    if deleted_file is None:
        raise HTTPException(status_code=404, detail=f"File with id={file_id} not found.")


def load_file_to_source_async(server: SyncServer, source_id: str, job_id: str, file: UploadFile, bytes: bytes):
    # write the file to a temporary directory (deleted after the context manager exits)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(str(tmpdirname), str(file.filename))
        with open(file_path, "wb") as buffer:
            buffer.write(bytes)

        server.load_file_to_source(source_id, file_path, job_id)
