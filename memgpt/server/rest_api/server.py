import importlib.util
import json
import logging
import os
import secrets
import subprocess
from typing import Optional

import typer
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from memgpt.server.constants import REST_DEFAULT_PORT
from memgpt.server.rest_api.auth.index import (
    setup_auth_router,  # TODO: probably remove right?
)
from memgpt.server.rest_api.interface import StreamingServerInterface
from memgpt.server.rest_api.routers.openai.assistants.assistants import (
    router as openai_assistants_router,
)
from memgpt.server.rest_api.routers.openai.assistants.threads import (
    router as openai_threads_router,
)
from memgpt.server.rest_api.routers.openai.chat_completions.chat_completions import (
    router as openai_chat_completions_router,
)
from memgpt.server.rest_api.routers.v1 import ROUTERS as v1_routes
from memgpt.server.rest_api.routers.v1.users import (
    router as users_router,  # TODO: decide on admin
)
from memgpt.server.server import SyncServer
from memgpt.settings import settings

"""
Basic REST API sitting on top of the internal MemGPT python server (SyncServer)

Start the server with:
  cd memgpt/server/rest_api
  poetry run uvicorn server:app --reload
"""

interface: StreamingServerInterface = StreamingServerInterface
server: SyncServer = SyncServer(default_interface_factory=lambda: interface())

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


ADMIN_PREFIX = "/v1/admin"
API_PREFIX = "/v1"
OPENAI_API_PREFIX = "/openai"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v1_routes are the MemGPT API routes
for route in v1_routes:
    app.include_router(route, prefix=API_PREFIX)
    # this gives undocumented routes for "latest" and bare api calls.
    # we should always tie this to the newest version of the api.
    app.include_router(route, prefix="", include_in_schema=False)
    app.include_router(route, prefix="/latest", include_in_schema=False)

# admin/users
app.include_router(users_router, prefix=ADMIN_PREFIX)

# openai
app.include_router(openai_assistants_router, prefix=OPENAI_API_PREFIX)
app.include_router(openai_threads_router, prefix=OPENAI_API_PREFIX)
app.include_router(openai_chat_completions_router, prefix=OPENAI_API_PREFIX)

# /api/auth endpoints
app.include_router(setup_auth_router(server, interface, password), prefix=API_PREFIX)

# Serve static files
static_files_path = os.path.join(os.path.dirname(importlib.util.find_spec("memgpt").origin), "server", "static_files")
app.mount("/assets", StaticFiles(directory=os.path.join(static_files_path, "assets")), name="static")


# Serve favicon
@app.get("/favicon.ico")
async def favicon():
    return FileResponse(os.path.join(static_files_path, "favicon.ico"))


# Middleware to handle API routes first
@app.middleware("http")
async def handle_api_routes(request: Request, call_next):
    if request.url.path.startswith(("/v1/", "/openai/")):
        response = await call_next(request)
        if response.status_code != 404:
            return response
    return await serve_spa(request.url.path)


# Catch-all route for SPA
async def serve_spa(full_path: str):
    return FileResponse(os.path.join(static_files_path, "index.html"))


# mount_static_files(app)


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
    if server:
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
