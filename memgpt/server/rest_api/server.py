import json
import logging
import os
import secrets
import subprocess
from typing import Optional

import typer
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.cors import CORSMiddleware

from memgpt.server.constants import REST_DEFAULT_PORT
from memgpt.server.rest_api.admin.users import setup_admin_router
from memgpt.server.rest_api.agents.command import setup_agents_command_router
from memgpt.server.rest_api.agents.config import setup_agents_config_router
from memgpt.server.rest_api.agents.index import setup_agents_index_router
from memgpt.server.rest_api.agents.memory import setup_agents_memory_router
from memgpt.server.rest_api.agents.message import setup_agents_message_router
from memgpt.server.rest_api.auth.index import setup_auth_router
from memgpt.server.rest_api.config.index import setup_config_index_router
from memgpt.server.rest_api.humans.index import setup_humans_index_router
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.models.index import setup_models_index_router
from memgpt.server.rest_api.openai_assistants.assistants import (
    setup_openai_assistant_router,
)
from memgpt.server.rest_api.personas.index import setup_personas_index_router
from memgpt.server.rest_api.presets.index import setup_presets_index_router
from memgpt.server.rest_api.sources.index import setup_sources_index_router
from memgpt.server.rest_api.static_files import mount_static_files
from memgpt.server.rest_api.tools.index import setup_tools_index_router
from memgpt.server.server import SyncServer
from memgpt.settings import settings

"""
Basic REST API sitting on top of the internal MemGPT python server (SyncServer)

Start the server with:
  cd memgpt/server/rest_api
  poetry run uvicorn server:app --reload
"""

interface: QueuingInterface = QueuingInterface()
server: SyncServer = SyncServer(default_interface=interface)

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /api/auth endpoints
app.include_router(setup_auth_router(server, interface, password), prefix=API_PREFIX)

# /admin/users endpoints
app.include_router(setup_admin_router(server, interface), prefix=ADMIN_PREFIX, dependencies=[Depends(verify_password)])

# /api/agents endpoints
app.include_router(setup_agents_command_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_agents_config_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_agents_index_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_agents_memory_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_agents_message_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_humans_index_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_personas_index_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_models_index_router(server, interface, password), prefix=API_PREFIX)
app.include_router(
    setup_tools_index_router(server, interface, password), prefix=API_PREFIX, dependencies=[Depends(verify_password)]
)  # admin only
app.include_router(setup_sources_index_router(server, interface, password), prefix=API_PREFIX)
app.include_router(setup_presets_index_router(server, interface, password), prefix=API_PREFIX)

# /api/config endpoints
app.include_router(setup_config_index_router(server, interface, password), prefix=API_PREFIX)

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
        app.openapi_schema["servers"] = [{"url": host} for host in settings.cors_origins]
        app.openapi_schema["info"]["title"] = "MemGPT API"

    # Split the API docs into MemGPT API, and OpenAI Assistants compatible API
    memgpt_api = app.openapi_schema.copy()
    memgpt_api["paths"] = {key: value for key, value in memgpt_api["paths"].items() if not key.startswith(OPENAI_API_PREFIX)}
    memgpt_api["info"]["title"] = "MemGPT API"
    with open("openapi_memgpt.json", "w") as file:
        print(f"Writing out openapi_memgpt.json file")
        json.dump(memgpt_api, file, indent=2)

    openai_assistants_api = app.openapi_schema.copy()
    openai_assistants_api["paths"] = {
        key: value
        for key, value in openai_assistants_api["paths"].items()
        if not (key.startswith(API_PREFIX) or key.startswith(ADMIN_PREFIX))
    }
    openai_assistants_api["info"]["title"] = "OpenAI Assistants API"
    with open("openapi_assistants.json", "w") as file:
        print(f"Writing out openapi_assistants.json file")
        json.dump(openai_assistants_api, file, indent=2)


@app.on_event("shutdown")
def on_shutdown():
    global server
    server.save_agents()
    server = None


def generate_self_signed_cert(cert_path="selfsigned.crt", key_path="selfsigned.key"):
    """Generate a self-signed SSL certificate.

    NOTE: intended to be used for development only.
    """
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:4096",
            "-keyout",
            key_path,
            "-out",
            cert_path,
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/C=US/ST=Denial/L=Springfield/O=Dis/CN=localhost",
        ],
        check=True,
    )
    return cert_path, key_path


def start_server(
    port: Optional[int] = None,
    host: Optional[str] = None,
    use_ssl: bool = False,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
    debug: bool = False,
):
    print("DEBUG", debug)
    if debug:
        from memgpt.server.server import logger as server_logger

        # Set the logging level
        server_logger.setLevel(logging.DEBUG)
        # Create a StreamHandler
        stream_handler = logging.StreamHandler()
        # Set the formatter (optional)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        # Add the handler to the logger
        server_logger.addHandler(stream_handler)

    if use_ssl:
        if ssl_cert is None:  # No certificate path provided, generate a self-signed certificate
            ssl_certfile, ssl_keyfile = generate_self_signed_cert()
            print(f"Running server with self-signed SSL cert: {ssl_certfile}, {ssl_keyfile}")
        else:
            ssl_certfile, ssl_keyfile = ssl_cert, ssl_key  # Assuming cert includes both
            print(f"Running server with provided SSL cert: {ssl_certfile}, {ssl_keyfile}")

        # This will start the server on HTTPS
        assert isinstance(ssl_certfile, str) and os.path.exists(ssl_certfile), ssl_certfile
        assert isinstance(ssl_keyfile, str) and os.path.exists(ssl_keyfile), ssl_keyfile
        print(
            f"Running: uvicorn server:app --host {host or 'localhost'} --port {port or REST_DEFAULT_PORT} --ssl-keyfile {ssl_keyfile} --ssl-certfile {ssl_certfile}"
        )
        uvicorn.run(
            app,
            host=host or "localhost",
            port=port or REST_DEFAULT_PORT,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
        )
    else:
        # Start the subprocess in a new session
        print(f"Running: uvicorn server:app --host {host or 'localhost'} --port {port or REST_DEFAULT_PORT}")
        uvicorn.run(
            app,
            host=host or "localhost",
            port=port or REST_DEFAULT_PORT,
        )
