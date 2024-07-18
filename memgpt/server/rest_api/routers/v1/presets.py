from typing import TYPE_CHECKING
from fastapi import APIRouter, Body, Depends, HTTPException, status, JSONResponse

from memgpt.server.rest_api.utils import get_current_actor, get_memgpt_server, get_current_interface
from memgpt.data_types import Preset
from memgpt.models.pydantic_models import PresetModel
from memgpt.server.schemas.presets import CreatePresetsRequest, ListPresetsResponse, CreatePresetResponse

if TYPE_CHECKING:
    from uuid import UUID
    from memgpt.orm.user import User
    from memgpt.server.server import SyncServer
    from memgpt.server.rest_api.interface import QueuingInterface

router = APIRouter(prefix="/presets", tags=["presets"])

@router.get("/{preset_name}",  response_model=PresetModel)
async def get_preset(
    preset_name: str,
    actor:"User" = Depends(get_current_actor),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """Get a preset."""
    return server.get_preset(user_id=actor._id, preset_name=preset_name)

@router.get("/",  response_model=ListPresetsResponse)
async def list_presets(
    actor:"User" = Depends(get_current_actor),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """List all presets created by a user."""
    # Clear the interface
    interface.clear()
    return ListPresetsResponse(presets=server.list_presets(user_id=actor._id))

@router.post("",  response_model=CreatePresetResponse)
async def create_preset(
    preset_request: CreatePresetsRequest,
    actor:"User" = Depends(get_current_actor),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """Create a preset."""
    # TODO server needs to return a preset schema, this is too many conversions
    # in the same chain for no reason
    return CreatePresetResponse(preset = server.create_preset(
        Preset(user_id=actor._id, **preset_request.model_dump(exclude_none=True))))


@router.delete("/{preset_id}", tags=["presets"])
async def delete_preset(
    preset_id: "UUID",
    actor:"User" = Depends(get_current_actor),
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """Delete a preset."""
    interface.clear()
    preset = server.delete_preset(user_id=actor._id, preset_id=preset_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"message": f"Preset preset_id={str(preset.id)} successfully deleted"}
    )
