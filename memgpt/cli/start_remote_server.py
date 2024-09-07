from typing import Optional

import uvicorn

from memgpt.server.rest_api.app import app
from memgpt.settings import settings


def start_server(
    port: Optional[int] = 8283,
    host: Optional[str] = "localhost",
    debug: bool = False,
):
    settings.debug = debug  # set debug logging for the application

    uvicorn.run(app, host=host, port=port)
