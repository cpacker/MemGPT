from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from letta.cli.cli import version
from letta.schemas.health import Health
from letta.server.rest_api.utils import get_letta_server
from letta.server.server import SyncServer

if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/health", tags=["health"])

# Health check
@router.get("/", response_model=Health, operation_id="health_check")
def health_check():
    return Health(
        version=version(),
        status="ok",
    )
