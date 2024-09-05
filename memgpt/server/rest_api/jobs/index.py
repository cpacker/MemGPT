from functools import partial
from typing import List

from fastapi import APIRouter, Depends

from memgpt.schemas.job import Job
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_jobs_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/jobs", tags=["jobs"], response_model=List[Job])
    async def list_jobs(
        user_id: str = Depends(get_current_user_with_server),
    ):
        interface.clear()

        # TODO: add filtering by status
        return server.list_jobs(user_id=user_id)

    @router.get("/jobs/active", tags=["jobs"], response_model=List[Job])
    async def list_active_jobs(
        user_id: str = Depends(get_current_user_with_server),
    ):
        interface.clear()
        return server.list_active_jobs(user_id=user_id)

    @router.get("/jobs/{job_id}", tags=["jobs"], response_model=Job)
    async def get_job(
        job_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        interface.clear()
        return server.get_job(job_id=job_id)

    return router
