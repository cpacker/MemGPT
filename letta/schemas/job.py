from datetime import datetime
from typing import Optional

from pydantic import Field

from letta.schemas.enums import JobStatus
from letta.schemas.letta_base import OrmMetadataBase


class JobBase(OrmMetadataBase):
    __id_prefix__ = "job"
    status: JobStatus = Field(default=JobStatus.created, description="The status of the job.")
    completed_at: Optional[datetime] = Field(None, description="The unix timestamp of when the job was completed.")
    metadata_: Optional[dict] = Field(None, description="The metadata of the job.")


class Job(JobBase):
    """
    Representation of offline jobs, used for tracking status of data loading tasks (involving parsing and embedding files).

    Parameters:
        id (str): The unique identifier of the job.
        status (JobStatus): The status of the job.
        created_at (datetime): The unix timestamp of when the job was created.
        completed_at (datetime): The unix timestamp of when the job was completed.
        user_id (str): The unique identifier of the user associated with the.

    """

    id: str = JobBase.generate_id_field()
    user_id: Optional[str] = Field(None, description="The unique identifier of the user associated with the job.")


class JobUpdate(JobBase):
    status: Optional[JobStatus] = Field(None, description="The status of the job.")

    class Config:
        extra = "ignore"  # Ignores extra fields
