import json

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from memgpt.settings import settings
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

    # TODO: refactor this
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


