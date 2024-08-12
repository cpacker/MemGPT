from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from memgpt.server.rest_api.utils import get_memgpt_server, get_current_interface

# these can be forward refs but because FastAPI uses them at runtime they need real imports
from uuid import UUID
from memgpt.orm.user import User
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface

router = APIRouter(prefix="/presets", tags=["presets"])

@router.get("/{preset_name}",  response_model=PresetModel)
async def get_preset(
    preset_name: str,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """Get a preset."""

    actor = server.get_current_user()
    return server.get_preset(user_id=actor._id, preset_name=preset_name)

@router.get("/",  response_model=ListPresetsResponse)
async def list_presets(
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """List all presets created by a user."""
    actor = server.get_current_user()
    # Clear the interface
    interface.clear()
    return ListPresetsResponse(presets=server.list_presets(user_id=actor._id))

@router.post("",  response_model=CreatePresetResponse)
async def create_preset(
    preset_request: CreatePresetsRequest,
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """Create a preset."""
    actor = server.get_current_user()
    # TODO server needs to return a preset schema, this is too many conversions
    # in the same chain for no reason
    return CreatePresetResponse(preset = server.create_preset(
        Preset(user_id=actor._id, **preset_request.model_dump(exclude_none=True))))


@router.delete("/{preset_id}", tags=["presets"])
async def delete_preset(
    preset_id: "UUID",
    interface: "QueuingInterface" = Depends(get_current_interface),
    server: "SyncServer" = Depends(get_memgpt_server),
):
    """Delete a preset."""
    actor = server.get_current_user()
    interface.clear()
    preset = server.delete_preset(user_id=actor._id, preset_id=preset_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK, content={"message": f"Preset preset_id={str(preset.id)} successfully deleted"}
    )
