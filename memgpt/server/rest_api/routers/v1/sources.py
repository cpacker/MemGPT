import tempfile
from typing import List

# These can be forward refs, but because Fastapi needs them at runtime the must be imported normally
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy.exc import MultipleResultsFound, NoResultFound

from memgpt.schemas.job import Job
from memgpt.schemas.source import Source, SourceCreate
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.server.schemas.sources import (
    GetSourceDocumentsResponse,
    GetSourcePassagesResponse,
)
from memgpt.server.server import SyncServer

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/{source_id}", response_model=Source)
async def get_source(
    source_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),
):
    """
    Get all sources
    """
    interface.clear()
    return server.get_source(source_id=source_id, user_id=server.get_current_user().id)


@router.get("/", response_model=List[Source])
async def list_sources(
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all data sources created by a user.
    """
    actor = server.get_current_user()
    interface.clear()
    return server.list_all_sources(user_id=actor.id)


@router.post("/", response_model=Source)
async def create_source(
    source: SourceCreate,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Create a new data source.
    """
    actor = server.get_current_user()
    interface.clear()
    return server.create_source(request=source, user_id=actor.id)


@router.delete("/{source_id}")
async def delete_source(
    source_id: "str",
    server: "SyncServer" = Depends(get_memgpt_server),
    interface: "QueuingInterface" = Depends(get_current_interface),
):
    """
    Delete a data source.
    """
    actor = server.get_current_user()
    interface.clear()
    server.delete_source(source_id=source_id, user_id=actor.id)


@router.post("/{source_id}/attach")
async def attach_source_to_agent(
    source_id: "str",
    agent_id: "str" = Query(..., description="The unique identifier of the agent to attach the source to."),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Attach a data source to an existing agent.
    """
    actor = server.get_current_user()
    interface.clear()
    source = server.ms.get_source(source_id=source_id, user_id=actor._id)
    source = server.attach_source_to_agent(source_name=source.name, agent_id=agent_id, user_id=actor._id)
    return Source(
        name=source.name,
        description=None,  # TODO: actually store descriptions
        user_id=source.user_id,
        id=source.id,
        embedding_config=server.server_embedding_config,
        created_at=source.created_at,
    )


@router.post("/{source_id}/detach")
async def detach_source_from_agent(
    source_id: "UUID",
    agent_id: "UUID" = Query(..., description="The unique identifier of the agent to detach the source from."),
    server: "SyncServer" = Depends(get_memgpt_server),
) -> None:
    """
    Detach a data source from an existing agent.
    """
    actor = server.get_current_user()
    server.detach_source_from_agent(source_id=source_id, agent_id=agent_id, user_id=actor._id)


@router.get("/status/{job_id}", response_model=Job)
async def get_job_status(
    job_id: "UUID",
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get the status of a job.
    """
    try:
        return server.ms.get_job(job_id=job_id)
    except (MultipleResultsFound, NoResultFound) as e:
        raise HTTPException(status_code=404, detail=f"Job with id={job_id} not found.") from e


@router.post("/{source_id}/upload", response_model=Job)
async def upload_file_to_source(
    file: UploadFile,
    source_id: "UUID",
    background_tasks: BackgroundTasks,
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Upload a file to a data source.
    """
    actor = server.get_current_user()
    interface.clear()
    source = server.ms.get_source(source_id=source_id, user_id=actor._id)
    bytes = file.file.read()

    # create job
    job = Job(user_id=actor._id, metadata={"type": "embedding", "filename": file.filename, "source_id": source_id})
    job_id = job.id
    server.ms.create_job(job)

    # create background task
    background_tasks.add_task(load_file_to_source_async, server, actor._id, source, job_id, file, bytes)

    # return job information
    job = server.ms.get_job(job_id=job_id)
    return job


@router.get("/{source_id}/passages")
async def list_passages(
    source_id: "UUID",
    server: SyncServer = Depends(get_memgpt_server),
):
    """
    List all passages associated with a data source.
    """
    actor = server.get_current_user()
    passages = server.list_data_source_passages(user_id=actor._id, source_id=source_id)
    return GetSourcePassagesResponse(passages=passages)


@router.get("/{source_id}/documents")
async def list_documents(
    source_id: "UUID",
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all documents associated with a data source.
    """
    actor = server.get_current_user()
    documents = server.list_data_source_documents(user_id=actor._id, source_id=source_id)
    return GetSourceDocumentsResponse(documents=documents)


def load_file_to_source_async(server: SyncServer, source_id: str, job_id: str, file: UploadFile, bytes: bytes):
    # write the file to a temporary directory (deleted after the context manager exits)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = os.path.join(tmpdirname, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(bytes)

        server.load_file_to_source(source_id, file_path, job_id)
