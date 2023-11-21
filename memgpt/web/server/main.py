import os
from os import getcwd

from starlette.middleware.cors import CORSMiddleware

from memgpt.config import MemGPTConfig
from memgpt.web.server.routers import sources, agents
from memgpt.web.server.routers.chat import setup_chat_ws_router

API_PREFIX = "/api"

CORS_ORIGINS = [
    "http://localhost:4200",
    "http://localhost:4201",
    "http://localhost:8000",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:4201",
    "http://127.0.0.1:8000",
]


def start_uvicorn_fastapi_server(config: MemGPTConfig):
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(setup_chat_ws_router(), prefix=API_PREFIX)
    app.include_router(sources.router, prefix=API_PREFIX)
    app.include_router(agents.router, prefix=API_PREFIX)

    mount_static_files(app)

    uvicorn.run(app, port=8000)


def mount_static_files(app):
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.staticfiles import StaticFiles
    from fastapi import HTTPException

    class SPAStaticFiles(StaticFiles):
        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except (HTTPException, StarletteHTTPException) as ex:
                if ex.status_code == 404:
                    return await super().get_response("index.html", scope)
                else:
                    raise ex

    app.mount(
        "/",
        SPAStaticFiles(
            directory=os.path.join(getcwd(), "memgpt", "web", "static_files"),
            html=True,
        ),
        name="spa-static-files",
    )
