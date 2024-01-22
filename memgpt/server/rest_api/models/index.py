import uuid
from typing import List
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface

router = APIRouter()


class Model(BaseModel):
    name: str = Field(..., description="The name of the model.")
    endpoint: str = Field(..., description="Endpoint URL for the model.")
    endpoint_type: str = Field(..., description="Type of the model endpoint.")
    wrapper: str = Field(None, description="Wrapper used for the model.")
    context_window: int = Field(..., description="Context window size for the model.")


class ListModelsResponse(BaseModel):
    models: List[Model] = Field(..., description="List of model configurations.")


def setup_models_index_router(server: SyncServer, interface: QueuingInterface):
    @router.get("/models", tags=["models"], response_model=ListModelsResponse)
    async def list_models(user_id: str = Query(..., description="Unique identifier of the user.")):
        # Validate and parse the user ID
        user_id = None if user_id == "null" else user_id
        user_id = uuid.UUID(user_id) if user_id else None

        # Clear the interface
        interface.clear()

        # TODO: Replace with actual data fetching logic once available
        models_data = [
            Model(
                name="ehartford/dolphin-2.5-mixtral-8x7b",
                endpoint="https://api.memgpt.ai",
                endpoint_type="vllm",
                wrapper="chatml",
                context_window=16384,
            ),
            Model(name="gpt-4", endpoint="https://api.openai.com/v1", endpoint_type="openai", context_window=8192),
        ]

        return ListModelsResponse(models=models_data)

    return router
