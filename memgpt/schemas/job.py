from datetime import datetime
from typing import Optional

from pydantic import Field

from memgpt.schemas.enums import JobStatus
from memgpt.schemas.memgpt_base import MemGPTBase
from memgpt.utils import get_utc_time


class JobBase(MemGPTBase):
    __id_prefix__ = "job"
    metadata_: Optional[dict] = Field({}, description="The metadata of the job.")


class Job(JobBase):
    """Representation of offline jobs."""

    id: str = JobBase.generate_id_field()
    status: JobStatus = Field(default=JobStatus.created, description="The status of the job.")
    created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the job was created.")
    completed_at: Optional[datetime] = Field(None, description="The unix timestamp of when the job was completed.")
    user_id: str = Field(..., description="The unique identifier of the user associated with the job.")


class JobUpdate(JobBase):
    id: str = Field(..., description="The unique identifier of the job.")
    status: Optional[JobStatus] = Field(..., description="The status of the job.")
