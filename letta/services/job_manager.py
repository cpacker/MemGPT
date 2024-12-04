from typing import List, Optional

from letta.orm.job import Job as JobModel
from letta.schemas.enums import JobStatus
from letta.schemas.job import Job as PydanticJob
from letta.schemas.job import JobUpdate
from letta.schemas.user import User as PydanticUser
from letta.utils import enforce_types, get_utc_time


class JobManager:
    """Manager class to handle business logic related to Jobs."""

    def __init__(self):
        # Fetching the db_context similarly as in OrganizationManager
        from letta.server.server import db_context

        self.session_maker = db_context

    @enforce_types
    def create_job(self, pydantic_job: PydanticJob, actor: PydanticUser) -> PydanticJob:
        """Create a new job based on the JobCreate schema."""
        with self.session_maker() as session:
            # Associate the job with the user
            pydantic_job.user_id = actor.id
            job_data = pydantic_job.model_dump()
            job = JobModel(**job_data)
            job.create(session, actor=actor)  # Save job in the database
        return job.to_pydantic()

    @enforce_types
    def update_job_by_id(self, job_id: str, job_update: JobUpdate, actor: PydanticUser) -> PydanticJob:
        """Update a job by its ID with the given JobUpdate object."""
        with self.session_maker() as session:
            # Fetch the job by ID
            job = JobModel.read(db_session=session, identifier=job_id)  # TODO: Add this later , actor=actor)

            # Update job attributes with only the fields that were explicitly set
            update_data = job_update.model_dump(exclude_unset=True, exclude_none=True)

            # Automatically update the completion timestamp if status is set to 'completed'
            if update_data.get("status") == JobStatus.completed and not job.completed_at:
                job.completed_at = get_utc_time()

            for key, value in update_data.items():
                setattr(job, key, value)

            # Save the updated job to the database
            return job.update(db_session=session)  # TODO: Add this later , actor=actor)

    @enforce_types
    def get_job_by_id(self, job_id: str, actor: PydanticUser) -> PydanticJob:
        """Fetch a job by its ID."""
        with self.session_maker() as session:
            # Retrieve job by ID using the Job model's read method
            job = JobModel.read(db_session=session, identifier=job_id)  # TODO: Add this later , actor=actor)
            return job.to_pydantic()

    @enforce_types
    def list_jobs(
        self, actor: PydanticUser, cursor: Optional[str] = None, limit: Optional[int] = 50, statuses: Optional[List[JobStatus]] = None
    ) -> List[PydanticJob]:
        """List all jobs with optional pagination and status filter."""
        with self.session_maker() as session:
            filter_kwargs = {"user_id": actor.id}

            # Add status filter if provided
            if statuses:
                filter_kwargs["status"] = statuses

            jobs = JobModel.list(
                db_session=session,
                cursor=cursor,
                limit=limit,
                **filter_kwargs,
            )
            return [job.to_pydantic() for job in jobs]

    @enforce_types
    def delete_job_by_id(self, job_id: str, actor: PydanticUser) -> PydanticJob:
        """Delete a job by its ID."""
        with self.session_maker() as session:
            job = JobModel.read(db_session=session, identifier=job_id)  # TODO: Add this later , actor=actor)
            job.hard_delete(db_session=session)  # TODO: Add this later , actor=actor)
            return job.to_pydantic()
