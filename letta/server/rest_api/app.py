import json
import logging
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from letta.__init__ import __version__
from letta.constants import ADMIN_PREFIX, API_PREFIX, OPENAI_API_PREFIX
from letta.schemas.letta_response import LettaResponse
from letta.server.constants import REST_DEFAULT_PORT

# NOTE(charles): these are extra routes that are not part of v1 but we still need to mount to pass tests
from letta.server.rest_api.auth.index import (
    setup_auth_router,  # TODO: probably remove right?
)
from letta.server.rest_api.interface import StreamingServerInterface
from letta.server.rest_api.routers.openai.assistants.assistants import (
    router as openai_assistants_router,
)
from letta.server.rest_api.routers.openai.assistants.threads import (
    router as openai_threads_router,
)
from letta.server.rest_api.routers.openai.chat_completions.chat_completions import (
    router as openai_chat_completions_router,
)

# from letta.orm.utilities import get_db_session  # TODO(ethan) reenable once we merge ORM
from letta.server.rest_api.routers.v1 import ROUTERS as v1_routes
from letta.server.rest_api.routers.v1.organizations import (
    router as organizations_router,
)
from letta.server.rest_api.routers.v1.users import (
    router as users_router,  # TODO: decide on admin
)
from letta.server.rest_api.static_files import mount_static_files
from letta.server.server import SyncServer
from letta.settings import settings

# TODO(ethan)
# NOTE(charles): @ethan I had to add this to get the global as the bottom to work
interface: StreamingServerInterface = StreamingServerInterface
server = SyncServer(default_interface_factory=lambda: interface())

# TODO: remove
password = None
## TODO(ethan): eventuall remove
# if password := settings.server_pass:
#    # if the pass was specified in the environment, use it
#    print(f"Using existing admin server password from environment.")
# else:
#    # Autogenerate a password for this session and dump it to stdout
#    password = secrets.token_urlsafe(16)
#    #typer.secho(f"Generated admin server password for this session: {password}", fg=typer.colors.GREEN)

import logging

from fastapi import FastAPI

log = logging.getLogger("uvicorn")


def generate_openapi_schema(app: FastAPI):
    # Update the OpenAPI schema
    if not app.openapi_schema:
        app.openapi_schema = app.openapi()

    openai_docs, letta_docs = [app.openapi_schema.copy() for _ in range(2)]

    openai_docs["paths"] = {k: v for k, v in openai_docs["paths"].items() if k.startswith("/openai")}
    openai_docs["info"]["title"] = "OpenAI Assistants API"
    letta_docs["paths"] = {k: v for k, v in letta_docs["paths"].items() if not k.startswith("/openai")}
    letta_docs["info"]["title"] = "Letta API"
    letta_docs["components"]["schemas"]["LettaResponse"] = {
        "properties": LettaResponse.model_json_schema(ref_template="#/components/schemas/LettaResponse/properties/{model}")["$defs"]
    }

    # Split the API docs into Letta API, and OpenAI Assistants compatible API
    for name, docs in [
        (
            "openai",
            openai_docs,
        ),
        (
            "letta",
            letta_docs,
        ),
    ]:
        if settings.cors_origins:
            docs["servers"] = [{"url": host} for host in settings.cors_origins]
        Path(f"openapi_{name}.json").write_text(json.dumps(docs, indent=2))


# middleware that only allows requests to pass through if user provides a password thats randomly generated and stored in memory
def generate_password():
    import secrets

    return secrets.token_urlsafe(16)


random_password = generate_password()


class CheckPasswordMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.headers.get("X-BARE-PASSWORD") == f"password {random_password}":
            return await call_next(request)

        return JSONResponse(
            content={"detail": "Unauthorized"},
            status_code=401,
        )


def create_application() -> "FastAPI":
    """the application start routine"""
    # global server
    # server = SyncServer(default_interface_factory=lambda: interface())
    print(f"\n[[ Letta server // v{__version__} ]]")

    app = FastAPI(
        swagger_ui_parameters={"docExpansion": "none"},
        # openapi_tags=TAGS_METADATA,
        title="Letta",
        summary="Create LLM agents with long-term memory and custom tools 📚🦙",
        version="1.0.0",  # TODO wire this up to the version in the package
        debug=True,
    )

    if "--ade" in sys.argv:
        settings.cors_origins.append("https://app.letta.com")
        print(f"▶ View using ADE at: https://app.letta.com/local-project/agents")

    if "--secure" in sys.argv:
        print(f"▶ Using secure mode with password: {random_password}")
        app.add_middleware(CheckPasswordMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for route in v1_routes:
        app.include_router(route, prefix=API_PREFIX)
        # this gives undocumented routes for "latest" and bare api calls.
        # we should always tie this to the newest version of the api.
        # app.include_router(route, prefix="", include_in_schema=False)
        app.include_router(route, prefix="/latest", include_in_schema=False)

    # NOTE: ethan these are the extra routes
    # TODO(ethan) remove

    # admin/users
    app.include_router(users_router, prefix=ADMIN_PREFIX)
    app.include_router(organizations_router, prefix=ADMIN_PREFIX)

    # openai
    app.include_router(openai_assistants_router, prefix=OPENAI_API_PREFIX)
    app.include_router(openai_threads_router, prefix=OPENAI_API_PREFIX)
    app.include_router(openai_chat_completions_router, prefix=OPENAI_API_PREFIX)

    # /api/auth endpoints
    app.include_router(setup_auth_router(server, interface, password), prefix=API_PREFIX)

    # / static files
    mount_static_files(app)

    @app.on_event("startup")
    def on_startup():
        # load the default tools
        # from letta.orm.tool import Tool

        # Tool.load_default_tools(get_db_session())

        generate_openapi_schema(app)

    @app.on_event("shutdown")
    def on_shutdown():
        global server
        server.save_agents()
        # server = None

    return app


app = create_application()


def start_server(
    port: Optional[int] = None,
    host: Optional[str] = None,
    debug: bool = False,
):
    """Convenience method to start the server from within Python"""
    if debug:
        from letta.server.server import logger as server_logger

        # Set the logging level
        server_logger.setLevel(logging.DEBUG)
        # Create a StreamHandler
        stream_handler = logging.StreamHandler()
        # Set the formatter (optional)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        stream_handler.setFormatter(formatter)
        # Add the handler to the logger
        server_logger.addHandler(stream_handler)

    print(f"▶ Server running at: http://{host or 'localhost'}:{port or REST_DEFAULT_PORT}\n")
    uvicorn.run(
        app,
        host=host or "localhost",
        port=port or REST_DEFAULT_PORT,
    )
