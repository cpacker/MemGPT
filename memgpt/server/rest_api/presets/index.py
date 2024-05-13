import uuid
from functools import partial
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA, DEFAULT_PRESET
from memgpt.data_types import Preset  # TODO remove
from memgpt.models.pydantic_models import HumanModel, PersonaModel, PresetModel
from memgpt.prompts import gpt_system
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
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
    id: Optional[str] = Field(None, description="The unique identifier of the preset.")
    # user_id: uuid.UUID = Field(..., description="The unique identifier of the user who created the preset.")
    description: Optional[str] = Field(None, description="The description of the preset.")
    # created_at: datetime = Field(default_factory=get_utc_time, description="The unix timestamp of when the preset was created.")
    system: Optional[str] = Field(None, description="The system prompt of the preset.")  # TODO: make optional and allow defaults
    persona: Optional[str] = Field(default=None, description="The persona of the preset.")
    human: Optional[str] = Field(default=None, description="The human of the preset.")
    functions_schema: List[Dict] = Field(..., description="The functions schema of the preset.")
    # TODO
    persona_name: Optional[str] = Field(None, description="The name of the persona of the preset.")
    human_name: Optional[str] = Field(None, description="The name of the human of the preset.")
    system_name: Optional[str] = Field(None, description="The name of the system prompt of the preset.")


class CreatePresetResponse(BaseModel):
    preset: PresetModel = Field(..., description="The newly created preset.")


def setup_presets_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/presets/{preset_name}", tags=["presets"], response_model=PresetModel)
    async def get_preset(
        preset_name: str,
        user_id: uuid.UUID = Depends(get_current_user_with_server),
    ):
        """Get a preset."""
        try:
            preset = server.get_preset(user_id=user_id, preset_name=preset_name)
            return preset
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

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

            # check if preset already exists
            # TODO: move this into a server function to create a preset
            if server.ms.get_preset(name=request.name, user_id=user_id):
                raise HTTPException(status_code=400, detail=f"Preset with name {request.name} already exists.")

            # For system/human/persona - if {system/human-personal}_name is None but the text is provied, then create a new data entry
            if not request.system_name and request.system:
                # new system provided without name identity
                system_name = f"system_{request.name}_{str(uuid.uuid4())}"
                system = request.system
                # TODO: insert into system table
            else:
                system_name = request.system_name if request.system_name else DEFAULT_PRESET
                system = request.system if request.system else gpt_system.get_system_text(system_name)

            if not request.human_name and request.human:
                # new human provided without name identity
                human_name = f"human_{request.name}_{str(uuid.uuid4())}"
                human = request.human
                server.ms.add_human(HumanModel(text=human, name=human_name, user_id=user_id))
            else:
                human_name = request.human_name if request.human_name else DEFAULT_HUMAN
                human = request.human if request.human else get_human_text(human_name)

            if not request.persona_name and request.persona:
                # new persona provided without name identity
                persona_name = f"persona_{request.name}_{str(uuid.uuid4())}"
                persona = request.persona
                server.ms.add_persona(PersonaModel(text=persona, name=persona_name, user_id=user_id))
            else:
                persona_name = request.persona_name if request.persona_name else DEFAULT_PERSONA
                persona = request.persona if request.persona else get_persona_text(persona_name)

            # create preset
            new_preset = Preset(
                user_id=user_id,
                id=request.id if request.id else uuid.uuid4(),
                name=request.name,
                description=request.description,
                system=system,
                persona=persona,
                persona_name=persona_name,
                human=human,
                human_name=human_name,
                functions_schema=request.functions_schema,
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
