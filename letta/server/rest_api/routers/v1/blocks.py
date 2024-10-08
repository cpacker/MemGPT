from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query

from letta.schemas.block import Block, CreateBlock, UpdateBlock
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer

if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/blocks", tags=["blocks"])


@router.get("/", response_model=List[Block], operation_id="list_memory_blocks")
def list_blocks(
    # query parameters
    label: Optional[str] = Query(None, description="Labels to include (e.g. human, persona)"),
    templates_only: bool = Query(True, description="Whether to include only templates"),
    name: Optional[str] = Query(None, description="Name of the block"),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = server.get_user_or_default(user_id=user_id)

    blocks = server.get_blocks(user_id=actor.id, label=label, template=templates_only, name=name)
    if blocks is None:
        return []
    return blocks


@router.post("/", response_model=Block, operation_id="create_memory_block")
def create_block(
    create_block: CreateBlock = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = server.get_user_or_default(user_id=user_id)

    create_block.user_id = actor.id
    return server.create_block(user_id=actor.id, request=create_block)


@router.patch("/{block_id}", response_model=Block, operation_id="update_memory_block")
def update_block(
    block_id: str,
    updated_block: UpdateBlock = Body(...),
    server: SyncServer = Depends(get_letta_server),
):
    # actor = server.get_current_user()

    updated_block.id = block_id
    return server.update_block(request=updated_block)


# TODO: delete should not return anything
@router.delete("/{block_id}", response_model=Block, operation_id="delete_memory_block")
def delete_block(
    block_id: str,
    server: SyncServer = Depends(get_letta_server),
):

    return server.delete_block(block_id=block_id)


@router.get("/{block_id}", response_model=Block, operation_id="get_memory_block")
def get_block(
    block_id: str,
    server: SyncServer = Depends(get_letta_server),
):

    block = server.get_block(block_id=block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")
    return block
