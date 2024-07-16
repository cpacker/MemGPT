import json
import secrets

import typer
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.cors import CORSMiddleware

from memgpt.server.rest_api.routers.v1 import ROUTERS as v1_routes

from memgpt.server.rest_api.admin.users import setup_admin_router
from memgpt.server.rest_api.agents.command import setup_agents_command_router
from memgpt.server.rest_api.agents.config import setup_agents_config_router
from memgpt.server.rest_api.agents.index import setup_agents_index_router
from memgpt.server.rest_api.agents.memory import setup_agents_memory_router
from memgpt.server.rest_api.agents.message import setup_agents_message_router
from memgpt.server.rest_api.auth.index import setup_auth_router
from memgpt.server.rest_api.config.index import setup_config_index_router
from memgpt.server.rest_api.humans.index import setup_humans_index_router
from memgpt.server.rest_api.models.index import setup_models_index_router
from memgpt.server.rest_api.openai_assistants.assistants import (
    setup_openai_assistant_router,
)
from memgpt.server.rest_api.openai_chat_completions.chat_completions import (
    setup_openai_chat_completions_router,
)
from memgpt.server.rest_api.personas.index import setup_personas_index_router
from memgpt.server.rest_api.presets.index import setup_presets_index_router
from memgpt.server.rest_api.sources.index import setup_sources_index_router
from memgpt.server.rest_api.static_files import mount_static_files
from memgpt.server.rest_api.tools.index import setup_user_tools_index_router
from memgpt.settings import settings

"""
Basic REST API sitting on top of the internal MemGPT python server (SyncServer)

Start the server with:
  cd memgpt/server/rest_api
  poetry run uvicorn server:app --reload
"""

if password := settings.server_pass:
    # if the pass was specified in the environment, use it
    print(f"Using existing admin server password from environment.")
else:
    # Autogenerate a password for this session and dump it to stdout
    password = secrets.token_urlsafe(16)
    typer.secho(f"Generated admin server password for this session: {password}", fg=typer.colors.GREEN)

security = HTTPBearer()


def verify_password(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """REST requests going to /admin are protected with a bearer token (that must match the password)"""
    if credentials.credentials != password:
        raise HTTPException(status_code=401, detail="Unauthorized")


ADMIN_PREFIX = "/admin"
API_PREFIX = "/api"
OPENAI_API_PREFIX = "/v1"

def create_application() -> "FastAPI":
    """the application start routine"""

    app = FastAPI(
        swagger_ui_parameters={"docExpansion": "none"},
        # openapi_tags=TAGS_METADATA,
        title="MemGPT",
        summary="Create LLM agents with long-term memory and custom tools 📚🦙",
        version="1.0.0",  # TODO wire this up to the version in the package
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for route in v1_routes:
        app.include_router(route, prefix="/v1")
        # this gives undocumented routes for "latest" and bare api calls.
        # we should always tie this to the newest version of the api.
        app.include_router(route, prefix="", include_in_schema=False)
        app.include_router(route, prefix="/latest", include_in_schema=False)
    return app

    # /api/auth endpoints
    #app.include_router(setup_auth_router(server, interface, password), prefix=API_PREFIX)

    ## /admin/users endpoints
    #app.include_router(setup_admin_router(server, interface), prefix=ADMIN_PREFIX, dependencies=[Depends(verify_password)])
    #app.include_router(setup_tools_index_router(server, interface), prefix=ADMIN_PREFIX, dependencies=[Depends(verify_password)])

    ## /api/agents endpoints
    #app.include_router(setup_agents_command_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_agents_config_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_agents_index_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_agents_memory_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_agents_message_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_humans_index_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_personas_index_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_models_index_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_user_tools_index_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_sources_index_router(server, interface, password), prefix=API_PREFIX)
    #app.include_router(setup_presets_index_router(server, interface, password), prefix=API_PREFIX)

    ## /api/config endpoints
    #app.include_router(setup_config_index_router(server, interface, password), prefix=API_PREFIX)

    ## /v1/assistants endpoints
    #app.include_router(setup_openai_assistant_router(server, interface), prefix=OPENAI_API_PREFIX)

    ## /v1/chat/completions endpoints
    #app.include_router(setup_openai_chat_completions_router(server, interface, password), prefix=OPENAI_API_PREFIX)

    # / static files
    mount_static_files(app)
    return app

app = create_application()

@app.on_event("startup")
def on_startup():
    # Update the OpenAPI schema
    if not app.openapi_schema:
        app.openapi_schema = app.openapi()

    if app.openapi_schema:
        app.openapi_schema["servers"] = [{"url": host} for host in settings.cors_origins]
        app.openapi_schema["info"]["title"] = "MemGPT API"

    # Split the API docs into MemGPT API, and OpenAI Assistants compatible API
    memgpt_api = app.openapi_schema.copy()
    memgpt_api["paths"] = {key: value for key, value in memgpt_api["paths"].items() if not key.startswith(OPENAI_API_PREFIX)}
    memgpt_api["info"]["title"] = "MemGPT API"
    with open("openapi_memgpt.json", "w", encoding="utf-8") as file:
        print(f"Writing out openapi_memgpt.json file")
        json.dump(memgpt_api, file, indent=2)

    openai_assistants_api = app.openapi_schema.copy()
    openai_assistants_api["paths"] = {
        key: value
        for key, value in openai_assistants_api["paths"].items()
        if not (key.startswith(API_PREFIX) or key.startswith(ADMIN_PREFIX))
    }
    openai_assistants_api["info"]["title"] = "OpenAI Assistants API"
    with open("openapi_assistants.json", "w", encoding="utf-8") as file:
        print(f"Writing out openapi_assistants.json file")
        json.dump(openai_assistants_api, file, indent=2)


@app.on_event("shutdown")
def on_shutdown():
    global server
    server.save_agents()
    server = None


