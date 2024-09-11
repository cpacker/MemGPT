from typing import List

from fastapi import APIRouter, Depends

from memgpt.schemas.job import Job
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.server import SyncServer

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=List[Job], operation_id="list_jobs")
def list_jobs(
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all jobs.
    """
    actor = server.get_current_user()

    # TODO: add filtering by status
    return server.list_jobs(user_id=actor.id)


@router.get("/active", response_model=List[Job], operation_id="list_active_jobs")
def list_active_jobs(
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    List all active jobs.
    """
    actor = server.get_current_user()

    return server.list_active_jobs(user_id=actor.id)


@router.get("/{job_id}", response_model=Job, operation_id="get_job")
def get_job(
    job_id: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """
    Get the status of a job.
    """

    return server.get_job(job_id=job_id)
