import json
import secrets
from pathlib import Path

import typer
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

# NOTE(charles): these are extra routes that are not part of v1 but we still need to mount to pass tests
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

# from memgpt.orm.utilities import get_db_session  # TODO(ethan) reenable once we merge ORM
from memgpt.server.rest_api.routers.v1 import ROUTERS as v1_routes
from memgpt.server.rest_api.routers.v1.users import (
    router as users_router,  # TODO: decide on admin
)
from memgpt.server.rest_api.static_files import mount_static_files
from memgpt.server.server import SyncServer
from memgpt.settings import settings

# TODO(ethan)
# NOTE(charles): @ethan I had to add this to get the global as the bottom to work
interface: StreamingServerInterface = StreamingServerInterface
server: SyncServer = SyncServer(default_interface_factory=lambda: interface())

# TODO(ethan): eventuall remove
if password := settings.server_pass:
    # if the pass was specified in the environment, use it
    print(f"Using existing admin server password from environment.")
else:
    # Autogenerate a password for this session and dump it to stdout
    password = secrets.token_urlsafe(16)
    typer.secho(f"Generated admin server password for this session: {password}", fg=typer.colors.GREEN)


ADMIN_PREFIX = "/v1/admin"
API_PREFIX = "/v1"
OPENAI_API_PREFIX = "/openai"


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
        app.include_router(route, prefix=API_PREFIX)
        # this gives undocumented routes for "latest" and bare api calls.
        # we should always tie this to the newest version of the api.
        app.include_router(route, prefix="", include_in_schema=False)
        app.include_router(route, prefix="/latest", include_in_schema=False)

    # NOTE: ethan these are the extra routes
    # TODO(ethan) remove

    # admin/users
    app.include_router(users_router, prefix=ADMIN_PREFIX)

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
        # from memgpt.orm.tool import Tool

        # Tool.load_default_tools(get_db_session())

        # Update the OpenAPI schema
        if not app.openapi_schema:
            app.openapi_schema = app.openapi()

        openai_docs, memgpt_docs = [app.openapi_schema.copy() for _ in range(2)]

        openai_docs["paths"] = {k: v for k, v in openai_docs["paths"].items() if k.startswith("/openai")}
        openai_docs["info"]["title"] = "OpenAI Assistants API"
        memgpt_docs["paths"] = {k: v for k, v in memgpt_docs["paths"].items() if not k.startswith("/openai")}
        memgpt_docs["info"]["title"] = "MemGPT API"

        # Split the API docs into MemGPT API, and OpenAI Assistants compatible API
        for name, docs in [
            (
                "openai",
                openai_docs,
            ),
            (
                "memgpt",
                memgpt_docs,
            ),
        ]:
            if settings.cors_origins:
                docs["servers"] = [{"url": host} for host in settings.cors_origins]
            Path(f"openapi_{name}.json").write_text(json.dumps(docs, indent=2))

    @app.on_event("shutdown")
    def on_shutdown():
        global server
        server.save_agents()
        server = None

    return app


app = create_application()
