import uuid
from functools import partial
from typing import List, Optional, Dict, Union

from fastapi import APIRouter, Body, Depends, Query, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memgpt.data_types import Preset  # TODO remove
from memgpt.models.pydantic_models import PresetModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from memgpt.utils import get_human_text, get_persona_text

router = APIRouter()

"""
Implement the following functions:
* List all available presets 
* Create a new preset 
* Delete a preset 
* TODO update a preset 
"""


class ListPresetsResponse(BaseModel):
    presets: List[PresetModel] = Field(..., description="List of available presets.")


class CreatePresetsRequest(BaseModel):
    # TODO is there a cleaner way to create the request from the PresetModel (need to drop fields though)?
    name: str = Field(..., description="The name of the preset.")
    id: Optional[Union[uuid.UUID, str]] = Field(default_factory=uuid.uuid4, description="The unique identifier of the preset.")
    # user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    # created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the preset was created.")
    system: str = Field(..., description="The system prompt of the preset.")
    persona: str = Field(default=get_persona_text(DEFAULT_PERSONA), description="The persona of the preset.")
    human: str = Field(default=get_human_text(DEFAULT_HUMAN), description="The human of the preset.")
    functions_schema: List[Dict] = Field(..., description="The functions schema of the preset.")
    # TODO
    persona_name: Optional[str] = Field(None, description="The name of the persona of the preset.")
    human_name: Optional[str] = Field(None, description="The name of the human of the preset.")


class CreatePresetResponse(BaseModel):
    preset: PresetModel = Field(..., description="The newly created preset.")


def setup_presets_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/presets", tags=["presets"], response_model=ListPresetsResponse)
    async def list_presets(
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """List all presets created by a user."""
        # Clear the interface
        interface.clear()

        try:
            presets = server.list_presets(user_id=user_id)
            return ListPresetsResponse(presets=presets)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.post("/presets", tags=["presets"], response_model=CreatePresetResponse)
    async def create_preset(
        request: CreatePresetsRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """Create a preset."""
        try:
            if isinstance(request.id, str):
                request.id = uuid.UUID(request.id)
            # new_preset = PresetModel(
            new_preset = Preset(
                user_id=user_id,
                id=request.id,
                name=request.name,
                description=request.description,
                system=request.system,
                persona=request.persona,
                human=request.human,
                functions_schema=request.functions_schema,
                persona_name=request.persona_name,
                human_name=request.human_name,
            )
            preset = server.create_preset(preset=new_preset)

            # TODO remove once we migrate from Preset to PresetModel
            preset = PresetModel(**vars(preset))

            return CreatePresetResponse(preset=preset)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.delete("/presets/{preset_id}", tags=["presets"])
    async def delete_preset(
        preset_id: uuid.UUID,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """Delete a preset."""
        interface.clear()
        try:
            preset = server.delete_preset(user_id=user_id, preset_id=preset_id)
            return JSONResponse(
                status_code=status.HTTP_200_OK, content={"message": f"Preset preset_id={str(preset.id)} successfully deleted"}
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    return router
