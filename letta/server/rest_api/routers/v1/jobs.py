from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query

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
    actor = server.get_user_or_default(user_id=user_id)

    # TODO: add filtering by status
    jobs = server.list_jobs(user_id=actor.id)

    # TODO: eventually use ORM
    # results = session.query(JobModel).filter(JobModel.user_id == user_id, JobModel.metadata_["source_id"].astext == sourced_id).all()
    if source_id:
        # can't be in the ORM since we have source_id stored in the metadata_
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
    actor = server.get_user_or_default(user_id=user_id)

    return server.list_active_jobs(user_id=actor.id)


@router.get("/{job_id}", response_model=Job, operation_id="get_job")
def get_job(
    job_id: str,
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Get the status of a job.
    """

    return server.get_job(job_id=job_id)


@router.delete("/{job_id}", response_model=Job, operation_id="delete_job")
def delete_job(
    job_id: str,
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Delete a job by its job_id.
    """
    job = server.get_job(job_id=job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    server.delete_job(job_id=job_id)
    return job
