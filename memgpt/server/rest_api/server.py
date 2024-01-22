import json
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


API_PREFIX = "/api"

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
