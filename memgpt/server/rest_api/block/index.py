from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from memgpt.schemas.block import Block, CreateBlock
from memgpt.schemas.block import Human as HumanModel  # TODO: modify
from memgpt.schemas.block import UpdateBlock
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()


def setup_block_index_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)

    @router.get("/blocks", tags=["block"], response_model=List[Block])
    async def list_blocks(
        user_id: str = Depends(get_current_user_with_server),
        # query parameters
        label: Optional[str] = Query(None, description="Labels to include (e.g. human, persona)"),
        templates_only: bool = Query(True, description="Whether to include only templates"),
        name: Optional[str] = Query(None, description="Name of the block"),
    ):
        # Clear the interface
        interface.clear()
        blocks = server.get_blocks(user_id=user_id, label=label, template=templates_only, name=name)
        if blocks is None:
            return []
        return blocks

    @router.post("/blocks", tags=["block"], response_model=Block)
    async def create_block(
        request: CreateBlock = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        interface.clear()
        request.user_id = user_id  # TODO: remove?
        return server.create_block(user_id=user_id, request=request)

    @router.post("/blocks/{block_id}", tags=["block"], response_model=Block)
    async def update_block(
        block_id: str,
        request: UpdateBlock = Body(...),
        user_id: str = Depends(get_current_user_with_server),
    ):
        interface.clear()
        # TODO: should this be in the param or the POST data?
        assert block_id == request.id
        return server.update_block(request)

    @router.delete("/blocks/{block_id}", tags=["block"], response_model=Block)
    async def delete_block(
        block_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        interface.clear()
        return server.delete_block(block_id=block_id)

    @router.get("/blocks/{block_id}", tags=["block"], response_model=Block)
    async def get_block(
        block_id: str,
        user_id: str = Depends(get_current_user_with_server),
    ):
        interface.clear()
        block = server.get_block(block_id=block_id)
        if block is None:
            raise HTTPException(status_code=404, detail="Block not found")
        return block

    return router
