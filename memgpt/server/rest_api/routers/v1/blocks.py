from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from memgpt.schemas.block import Block, CreateBlock, UpdateBlock
from memgpt.server.rest_api.utils import get_memgpt_server
from memgpt.server.server import SyncServer

if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/blocks", tags=["blocks"])


@router.get("/", response_model=List[Block])
async def list_blocks(
    # query parameters
    label: Optional[str] = Query(None, description="Labels to include (e.g. human, persona)"),
    templates_only: bool = Query(True, description="Whether to include only templates"),
    name: Optional[str] = Query(None, description="Name of the block"),
    server: SyncServer = Depends(get_memgpt_server),
):
    actor = server.get_current_user()

    blocks = server.get_blocks(user_id=actor.id, label=label, template=templates_only, name=name)
    if blocks is None:
        return []
    return blocks


@router.post("/", response_model=Block)
async def create_block(
    create_block: CreateBlock = Body(...),
    server: SyncServer = Depends(get_memgpt_server),
):
    actor = server.get_current_user()

    create_block.user_id = actor.id
    return server.create_block(user_id=actor.id, request=create_block)


@router.patch("/{block_id}", response_model=Block)
async def update_block(
    block_id: str,
    updated_block: UpdateBlock = Body(...),
    server: SyncServer = Depends(get_memgpt_server),
):
    # actor = server.get_current_user()

    updated_block.id = block_id
    return server.update_block(request=updated_block)


# TODO: delete should not return anything
@router.delete("/{block_id}", response_model=Block)
async def delete_block(
    block_id: str,
    server: SyncServer = Depends(get_memgpt_server),
):

    return server.delete_block(block_id=block_id)


@router.get("/{block_id}", response_model=Block)
async def get_block(
    block_id: str,
    server: SyncServer = Depends(get_memgpt_server),
):

    block = server.get_block(block_id=block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")
    return block
