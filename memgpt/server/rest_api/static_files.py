import importlib.util
import os

from fastapi import FastAPI, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                return await super().get_response("index.html", scope)
            else:
                raise ex


def mount_static_files(app: FastAPI):
    static_files_path = os.path.join(os.path.dirname(importlib.util.find_spec("memgpt").origin), "server", "static_files")
    if os.path.exists(static_files_path):
        app.mount(
            "/",
            # "/app",
            SPAStaticFiles(
                directory=static_files_path,
                html=True,
            ),
            name="spa-static-files",
        )


# def mount_static_files(app: FastAPI):
#     static_files_path = os.path.join(os.path.dirname(importlib.util.find_spec("memgpt").origin), "server", "static_files")
#     if os.path.exists(static_files_path):

#         @app.get("/{full_path:path}")
#         async def serve_spa(full_path: str):
#             if full_path.startswith("v1"):
#                 raise HTTPException(status_code=404, detail="Not found")
#             file_path = os.path.join(static_files_path, full_path)
#             if os.path.isfile(file_path):
#                 return FileResponse(file_path)
#             return FileResponse(os.path.join(static_files_path, "index.html"))
