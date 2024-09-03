import json
from pathlib import Path

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from memgpt.settings import settings
from memgpt.orm.utilities import get_db_session
from memgpt.server.rest_api.static_files import mount_static_files
from memgpt.server.rest_api.routers.v1 import ROUTERS as v1_routes


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

    # / static files
    mount_static_files(app)

    @app.on_event("startup")
    def on_startup():
        # load the default tools
        from memgpt.orm.tool import Tool
        Tool.load_default_tools(get_db_session())

        # Update the OpenAPI schema
        if not app.openapi_schema:
            app.openapi_schema = app.openapi()

        openai_docs, memgpt_docs = [app.openapi_schema.copy() for _ in range(2)]

        openai_docs["paths"] = {k:v for k,v in openai_docs["paths"].items() if k.startswith("/openai")}
        openai_docs["info"]["title"] = "OpenAI Assistants API"
        memgpt_docs["paths"] = {k:v for k,v in memgpt_docs["paths"].items() if not k.startswith("/openai")}
        memgpt_docs["info"]["title"] = "MemGPT API"

        # Split the API docs into MemGPT API, and OpenAI Assistants compatible API
        for name, docs in [("openai", openai_docs,), ("memgpt", memgpt_docs,)]:
            docs["servers"] = [{"url": host} for host in settings.cors_origins]
            Path(f"openapi_{name}.json").write_text(json.dumps(docs, indent=2))

    @app.on_event("shutdown")
    def on_shutdown():
        global server
        server.save_agents()
        server = None

    return app

app = create_application()
