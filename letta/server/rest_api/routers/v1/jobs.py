from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from letta.orm.errors import NoResultFound
from letta.schemas.enums import JobStatus
from letta.schemas.job import Job
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/", response_model=List[Job], operation_id="list_jobs")
def list_jobs(
    server: "SyncServer" = Depends(get_letta_server),
    source_id: Optional[str] = Query(None, description="Only list jobs associated with the source."),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    List all jobs.
    """
    actor = server.user_manager.get_user_or_default(user_id=user_id)

    # TODO: add filtering by status
    jobs = server.job_manager.list_jobs(actor=actor)

    if source_id:
        # can't be in the ORM since we have source_id stored in the metadata_
        # TODO: Probably change this
        jobs = [job for job in jobs if job.metadata_.get("source_id") == source_id]
    return jobs


@router.get("/active", response_model=List[Job], operation_id="list_active_jobs")
def list_active_jobs(
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    List all active jobs.
    """
    actor = server.user_manager.get_user_or_default(user_id=user_id)

    return server.job_manager.list_jobs(actor=actor, statuses=[JobStatus.created, JobStatus.running])


@router.get("/{job_id}", response_model=Job, operation_id="get_job")
def get_job(
    job_id: str,
    user_id: Optional[str] = Header(None, alias="user_id"),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Get the status of a job.
    """
    actor = server.user_manager.get_user_or_default(user_id=user_id)

    try:
        return server.job_manager.get_job_by_id(job_id=job_id, actor=actor)
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Job not found")


@router.delete("/{job_id}", response_model=Job, operation_id="delete_job")
def delete_job(
    job_id: str,
    user_id: Optional[str] = Header(None, alias="user_id"),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Delete a job by its job_id.
    """
    actor = server.user_manager.get_user_or_default(user_id=user_id)

    try:
        job = server.job_manager.delete_job_by_id(job_id=job_id, actor=actor)
        return job
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Job not found")
