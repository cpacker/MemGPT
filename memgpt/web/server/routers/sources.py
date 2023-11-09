from fastapi import APIRouter

from memgpt.connectors.storage import StorageConnector

router = APIRouter()


@router.get("/sources", tags=["sources"])
async def available_sources():
    return StorageConnector.list_loaded_data()
