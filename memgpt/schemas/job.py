import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from memgpt.utils import get_utc_time


class JobStatus(str, Enum):
    created = "created"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobModel(BaseModel, table=True):
    """Representation of offline jobs."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the job.", primary_key=True)
    status: JobStatus = Field(default=JobStatus.created, description="The status of the job.")
    created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the job was created.")
    completed_at: Optional[datetime] = Field(None, description="The unix timestamp of when the job was completed.")
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the job.")
    metadata_: Optional[dict] = Field({}, description="The metadata of the job.")


class JobCreate(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="The unique identifier of the job.", primary_key=True)
    user_id: uuid.UUID = Field(..., description="The unique identifier of the user associated with the job.")
    metadata_: Optional[dict] = Field({}, description="The metadata of the job.")


class JobUpdate(BaseModel):
    job_id: uuid.UUID = Field(..., description="The unique identifier of the job.")
    status: Optional[JobStatus] = Field(..., description="The status of the job.")
    metadata_: Optional[dict] = Field({}, description="The metadata of the job.")
