from typing import List

from fastapi import APIRouter, Body, HTTPException, Path, Query

from letta.constants import DEFAULT_PRESET
from letta.schemas.openai.openai import AssistantFile, OpenAIAssistant
from letta.server.rest_api.routers.openai.assistants.schemas import (
    CreateAssistantFileRequest,
    CreateAssistantRequest,
    DeleteAssistantFileResponse,
    DeleteAssistantResponse,
)
from letta.utils import get_utc_time

router = APIRouter()


# TODO: implement mechanism for creating/authenticating users associated with a bearer token
router = APIRouter(prefix="/v1/assistants", tags=["assistants"])


# create assistant (Letta agent)
@router.post("/", response_model=OpenAIAssistant)
def create_assistant(request: CreateAssistantRequest = Body(...)):
    # TODO: create preset
    return OpenAIAssistant(
        id=DEFAULT_PRESET,
        name="default_preset",
        description=request.description,
        created_at=int(get_utc_time().timestamp()),
        model=request.model,
        instructions=request.instructions,
        tools=request.tools,
        file_ids=request.file_ids,
        metadata=request.metadata,
    )


@router.post("/{assistant_id}/files", response_model=AssistantFile)
def create_assistant_file(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    request: CreateAssistantFileRequest = Body(...),
):
    # TODO: add file to assistant
    return AssistantFile(
        id=request.file_id,
        created_at=int(get_utc_time().timestamp()),
        assistant_id=assistant_id,
    )


@router.get("/", response_model=List[OpenAIAssistant])
def list_assistants(
    limit: int = Query(1000, description="How many assistants to retrieve."),
    order: str = Query("asc", description="Order of assistants to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: implement list assistants (i.e. list available Letta presets)
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/{assistant_id}/files", response_model=List[AssistantFile])
def list_assistant_files(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    limit: int = Query(1000, description="How many files to retrieve."),
    order: str = Query("asc", description="Order of files to retrieve (either 'asc' or 'desc')."),
    after: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
    before: str = Query(None, description="A cursor for use in pagination. `after` is an object ID that defines your place in the list."),
):
    # TODO: list attached data sources to preset
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/{assistant_id}", response_model=OpenAIAssistant)
def retrieve_assistant(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
):
    # TODO: get and return preset
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.get("/{assistant_id}/files/{file_id}", response_model=AssistantFile)
def retrieve_assistant_file(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    file_id: str = Path(..., description="The unique identifier of the file."),
):
    # TODO: return data source attached to preset
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.post("/{assistant_id}", response_model=OpenAIAssistant)
def modify_assistant(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    request: CreateAssistantRequest = Body(...),
):
    # TODO: modify preset
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.delete("/{assistant_id}", response_model=DeleteAssistantResponse)
def delete_assistant(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
):
    # TODO: delete preset
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")


@router.delete("/{assistant_id}/files/{file_id}", response_model=DeleteAssistantFileResponse)
def delete_assistant_file(
    assistant_id: str = Path(..., description="The unique identifier of the assistant."),
    file_id: str = Path(..., description="The unique identifier of the file."),
):
    # TODO: delete source on preset
    raise HTTPException(status_code=404, detail="Not yet implemented (coming soon)")
