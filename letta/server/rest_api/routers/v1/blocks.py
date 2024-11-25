from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query

from letta.orm.errors import NoResultFound
from letta.schemas.block import Block, BlockUpdate, CreateBlock
from letta.schemas.memory import Memory
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
    create_block: CreateBlock = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    actor = server.get_user_or_default(user_id=user_id)
    block = Block(**create_block.model_dump())
    return server.block_manager.create_or_update_block(actor=actor, block=block)


@router.patch("/{block_id}", response_model=Block, operation_id="update_memory_block")
def update_block(
    block_id: str,
    update_block: BlockUpdate = Body(...),
    server: SyncServer = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),
):
    actor = server.get_user_or_default(user_id=user_id)
    return server.block_manager.update_block(block_id=block_id, block_update=update_block, actor=actor)


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
    print("call get block", block_id)
    actor = server.get_user_or_default(user_id=user_id)
    try:
        block = server.block_manager.get_block_by_id(block_id=block_id, actor=actor)
        if block is None:
            raise HTTPException(status_code=404, detail="Block not found")
        return block
    except NoResultFound:
        raise HTTPException(status_code=404, detail="Block not found")


@router.patch("/{block_id}/attach", response_model=Block, operation_id="update_agent_memory_block")
def link_agent_memory_block(
    block_id: str,
    agent_id: str = Query(..., description="The unique identifier of the agent to attach the source to."),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Link a memory block to an agent.
    """
    actor = server.get_user_or_default(user_id=user_id)

    block = server.block_manager.get_block_by_id(block_id=block_id, actor=actor)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")

    server.blocks_agents_manager.add_block_to_agent(agent_id=agent_id, block_id=block_id, block_label=block.label)
    return block


@router.patch("/{block_id}/detach", response_model=Memory, operation_id="update_agent_memory_block")
def unlink_agent_memory_block(
    block_id: str,
    agent_id: str = Query(..., description="The unique identifier of the agent to attach the source to."),
    server: "SyncServer" = Depends(get_letta_server),
    user_id: Optional[str] = Header(None, alias="user_id"),  # Extract user_id from header, default to None if not present
):
    """
    Unlink a memory block from an agent
    """
    actor = server.get_user_or_default(user_id=user_id)

    block = server.block_manager.get_block_by_id(block_id=block_id, actor=actor)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")
    # Link the block to the agent
    server.blocks_agents_manager.remove_block_with_id_from_agent(agent_id=agent_id, block_id=block_id)
    return block
