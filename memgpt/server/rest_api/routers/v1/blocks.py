from typing import TYPE_CHECKING, List, Optional
from fastapi import APIRouter, Body, Depends, HTTPException
from memgpt.server.schemas.tools import CreateToolRequest, ListToolsResponse
from memgpt.server.rest_api.utils import get_current_interface, get_memgpt_server
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer
from memgpt.schemas.block import Block, UpdateBlock, CreateBlock

if TYPE_CHECKING:
    from memgpt.orm.user import User

router = APIRouter(prefix="/blocks", tags=["blocks"])

@router.get("/", response_model=List[Block])
async def list_blocks(
    # query parameters
    label: Optional[str] = Query(None, description="Labels to include (e.g. human, persona)"),
    templates_only: bool = Query(True, description="Whether to include only templates"),
    name: Optional[str] = Query(None, description="Name of the block"),
    server: SyncServer = Depends(get_memgpt_server),
    interface: QueuingInterface = Depends(get_current_interface),
):
    # Clear the interface
    interface.clear()
    user = server.get_current_user()
    blocks = server.get_blocks(user_id=user.id, label=label, template=templates_only, name=name)
    if blocks is None:
        return []
    return blocks

@router.post("/", response_model=Block)
async def create_block(
    create_block: CreateBlock = Body(...),
    server: SyncServer = Depends(get_memgpt_server),
    interface: QueuingInterface = Depends(get_current_interface),
):
    interface.clear()
    user = server.get_current_user()
    return server.create_block(user_id=user.id, request=create_block)

@router.post("/{block_id}", response_model=Block)
async def update_block(
    block_id: str,
    updated_block: UpdateBlock = Body(...),
    server: SyncServer = Depends(get_memgpt_server),
    interface: QueuingInterface = Depends(get_current_interface),
):

    interface.clear()
    updated_block.id = block_id
    user = server.get_current_user()
    return server.update_block(user_id=user.id, request=updated_block)

@router.delete("/{block_id}", response_model=Block)
async def delete_block(
    block_id: str,
    server: SyncServer = Depends(get_memgpt_server),
    interface: QueuingInterface = Depends(get_current_interface),
):
    interface.clear()
    return server.delete_block(block_id=block_id)

@router.get("/{block_id}", response_model=Block)
async def get_block(
    block_id: str,
    server: SyncServer = Depends(get_memgpt_server),
    interface: QueuingInterface = Depends(get_current_interface),
):
    interface.clear()
    block = server.get_block(block_id=block_id)
    if block is None:
        raise HTTPException(status_code=404, detail="Block not found")
    return block


