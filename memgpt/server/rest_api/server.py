import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware

from memgpt.constants import JSON_ENSURE_ASCII
from memgpt.server.rest_api.agents.index import setup_agents_index_router
from memgpt.server.rest_api.agents.command import setup_agents_command_router
from memgpt.server.rest_api.agents.config import setup_agents_config_router
from memgpt.server.rest_api.agents.memory import setup_agents_memory_router
from memgpt.server.rest_api.agents.message import setup_agents_message_router
from memgpt.server.rest_api.auth.index import setup_auth_router
from memgpt.server.rest_api.config.index import setup_config_index_router
from memgpt.server.rest_api.humans.index import setup_humans_index_router
from memgpt.server.rest_api.personas.index import setup_personas_index_router
from memgpt.server.rest_api.models.index import setup_models_index_router
from memgpt.server.rest_api.openai_assistants.assistants import setup_openai_assistant_router
from memgpt.server.rest_api.admin.users import setup_admin_router
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.static_files import mount_static_files

"""
Basic REST API sitting on top of the internal MemGPT python server (SyncServer)

Start the server with:
  cd memgpt/server/rest_api
  poetry run uvicorn server:app --reload
"""

interface: QueuingInterface = QueuingInterface()
server: SyncServer = SyncServer(default_interface=interface)

# TODO remove, hack for now to set up an init API key for testing
# new_key = server.create_api_key_for_user(user_id=uuid.UUID("00000000000000000000a61b692e9d3d"))
# print(f"new_key = {new_key.token}")

ADMIN_PREFIX = "/admin"
API_PREFIX = "/api"
OPENAI_API_PREFIX = "/v1"

CORS_ORIGINS = [
    "http://localhost:4200",
    "http://localhost:4201",
    "http://localhost:8283",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:4201",
    "http://127.0.0.1:8283",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# /api/auth endpoints
app.include_router(setup_auth_router(server, interface), prefix=API_PREFIX)
# /admin/users endpoints
app.include_router(setup_admin_router(server, interface), prefix=ADMIN_PREFIX)
# /api/agents endpoints
app.include_router(setup_agents_command_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_config_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_index_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_memory_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_message_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_humans_index_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_personas_index_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_models_index_router(server, interface), prefix=API_PREFIX)
# /api/config endpoints
app.include_router(setup_config_index_router(server, interface), prefix=API_PREFIX)
# /v1/assistants endpoints
app.include_router(setup_openai_assistant_router(server, interface), prefix=OPENAI_API_PREFIX)
# / static files
mount_static_files(app)


@app.on_event("startup")
def on_startup():
    # Update the OpenAPI schema
    if not app.openapi_schema:
        app.openapi_schema = app.openapi()

    if app.openapi_schema:
        app.openapi_schema["servers"] = [{"url": "http://localhost:8283"}]
        app.openapi_schema["info"]["title"] = "MemGPT API"

    # Write out the OpenAPI schema to a file
    with open("openapi.json", "w") as file:
        print(f"Writing out openapi.json file")
        json.dump(app.openapi_schema, file, indent=2)


@app.on_event("shutdown")
def on_shutdown():
    global server
    server.save_agents()
    server = None
