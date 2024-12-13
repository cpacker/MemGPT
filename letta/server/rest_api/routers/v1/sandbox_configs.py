from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from letta.schemas.sandbox_config import SandboxConfig as PydanticSandboxConfig
from letta.schemas.sandbox_config import SandboxConfigCreate, SandboxConfigUpdate
from letta.schemas.sandbox_config import SandboxEnvironmentVariable as PydanticEnvVar
from letta.schemas.sandbox_config import (
    SandboxEnvironmentVariableCreate,
    SandboxEnvironmentVariableUpdate,
    SandboxType,
)
from letta.server.rest_api.utils import get_letta_server, get_user_id
from letta.server.server import SyncServer

router = APIRouter(prefix="/sandbox-config", tags=["sandbox-config"])


### Sandbox Config Routes


@router.post("/", response_model=PydanticSandboxConfig)
def create_sandbox_config(
    config_create: SandboxConfigCreate,
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)

    return server.sandbox_config_manager.create_or_update_sandbox_config(config_create, actor)


@router.post("/e2b/default", response_model=PydanticSandboxConfig)
def create_default_e2b_sandbox_config(
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    return server.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.E2B, actor=actor)


@router.post("/local/default", response_model=PydanticSandboxConfig)
def create_default_local_sandbox_config(
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    return server.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.LOCAL, actor=actor)


@router.patch("/{sandbox_config_id}", response_model=PydanticSandboxConfig)
def update_sandbox_config(
    sandbox_config_id: str,
    config_update: SandboxConfigUpdate,
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    return server.sandbox_config_manager.update_sandbox_config(sandbox_config_id, config_update, actor)


@router.delete("/{sandbox_config_id}", status_code=204)
def delete_sandbox_config(
    sandbox_config_id: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    server.sandbox_config_manager.delete_sandbox_config(sandbox_config_id, actor)


@router.get("/", response_model=List[PydanticSandboxConfig])
def list_sandbox_configs(
    limit: int = Query(1000, description="Number of results to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor to fetch the next set of results"),
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    return server.sandbox_config_manager.list_sandbox_configs(actor, limit=limit, cursor=cursor)


### Sandbox Environment Variable Routes


@router.post("/{sandbox_config_id}/environment-variable", response_model=PydanticEnvVar)
def create_sandbox_env_var(
    sandbox_config_id: str,
    env_var_create: SandboxEnvironmentVariableCreate,
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    return server.sandbox_config_manager.create_sandbox_env_var(env_var_create, sandbox_config_id, actor)


@router.patch("/environment-variable/{env_var_id}", response_model=PydanticEnvVar)
def update_sandbox_env_var(
    env_var_id: str,
    env_var_update: SandboxEnvironmentVariableUpdate,
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    return server.sandbox_config_manager.update_sandbox_env_var(env_var_id, env_var_update, actor)


@router.delete("/environment-variable/{env_var_id}", status_code=204)
def delete_sandbox_env_var(
    env_var_id: str,
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    server.sandbox_config_manager.delete_sandbox_env_var(env_var_id, actor)


@router.get("/{sandbox_config_id}/environment-variable", response_model=List[PydanticEnvVar])
def list_sandbox_env_vars(
    sandbox_config_id: str,
    limit: int = Query(1000, description="Number of results to return"),
    cursor: Optional[str] = Query(None, description="Pagination cursor to fetch the next set of results"),
    server: SyncServer = Depends(get_letta_server),
    user_id: str = Depends(get_user_id),
):
    actor = server.user_manager.get_user_or_default(user_id=user_id)
    return server.sandbox_config_manager.list_sandbox_env_vars(sandbox_config_id, actor, limit=limit, cursor=cursor)
