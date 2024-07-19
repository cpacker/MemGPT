import re
import uuid
from functools import partial
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from memgpt.models.pydantic_models import (
    AgentStateModel,
    EmbeddingConfigModel,
    LLMConfigModel,
)
from memgpt.server.rest_api.auth_token import get_current_user
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.server import SyncServer

router = APIRouter()

def setup_agents_config_router(server: SyncServer, interface: QueuingInterface, password: str):
    get_current_user_with_server = partial(partial(get_current_user, server), password)


    return router
