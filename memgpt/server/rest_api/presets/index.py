import uuid
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, Query, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import PresetModel
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

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
    preset: PresetModel = Field(..., description="The name of the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")


class CreateSourceResponse(BaseModel):
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
            presets = server.ms.list_presets(user_id=user_id)
            return ListPresetsResponse(presets=presets)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    @router.post("/presets", tags=["presets"], response_model=PresetModel)
    async def create_preset(
        request: CreatePresetsRequest = Body(...),
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """Create a preset."""
        try:
            new_preset = PresetModel(**vars(request))
            server.ms.create_preset(preset=new_preset)
            return new_preset
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
            # first get the preset by name
            preset = server.get_preset(preset_id=preset_id, user_id=user_id)
            if preset is None:
                raise ValueError(f"Could not find preset_id {preset_id}")
            # then delete via name
            # TODO allow delete-by-id, eg via server.delete_preset function
            server.ms.delete_preset(name=preset.name, user_id=user_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content={"message": f"Preset preset_id={preset_id} successfully deleted"})
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
