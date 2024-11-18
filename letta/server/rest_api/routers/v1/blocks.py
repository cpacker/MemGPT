from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query

from letta.orm.errors import NoResultFound
from letta.schemas.block import Block, BlockCreate, BlockUpdate
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
    return server.block_manager.get_blocks(actor=actor, label=label, is_template=templates_only, template_name=name)


@router.post("/", response_model=Block, operation_id="create_memory_block")
def create_block(
    create_block: BlockCreate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = server.get_user_or_default(user_id=user_id)
    block = Block(**create_block.model_dump())
    return server.block_manager.create_or_update_block(actor=actor, block=block)


@router.patch("/{block_id}", response_model=Block, operation_id="update_memory_block")
def update_block(
    block_id: str,
    updated_block: BlockUpdate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    actor = server.get_user_or_default(user_id=user_id)
    return server.block_manager.update_block(block_id=block_id, block_update=updated_block, actor=actor)


@router.delete("/{block_id}", response_model=Block, operation_id="delete_memory_block")
def delete_block(
    block_id: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    actor = server.get_user_or_default(user_id=user_id)
    return server.block_manager.delete_block(block_id=block_id, actor=actor)


@router.get("/{block_id}", response_model=Block, operation_id="get_memory_block")
def get_block(
    block_id: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    actor = server.get_user_or_default(user_id=user_id)
    try:
        return server.block_manager.get_block_by_id(block_id=block_id, actor=actor)
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Block not found")
