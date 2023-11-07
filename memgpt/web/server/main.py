import datetime
import os
from copy import deepcopy
from os import getcwd

from memgpt.agent import AgentAsync
from memgpt.config import MemGPTConfig
from memgpt.utils import parse_json


def parse_function_call(message):
    new_message = deepcopy(message)
    if new_message["role"] == "assistant" and new_message["function_call"]["name"] == "send_message":
        new_message["function_call"]["arguments"] = parse_json(new_message["function_call"]["arguments"])
    return new_message


def map_messages_to_parsed_function_call_arguments(all_messages: list):
    return list(map(parse_function_call, all_messages))


def start_uvicorn_fastapi_server(agent: AgentAsync, config: MemGPTConfig):
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()

    setup_endpoints(app, agent, config)

    uvicorn.run(app, port=8000)


def setup_endpoints(app, memgpt_agent: AgentAsync, config: MemGPTConfig):
    from starlette.websockets import WebSocket
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

    @app.websocket("/api/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            user_message = await websocket.receive_text()

            (
                new_messages,
                heartbeat_request,
                function_failed,
                token_warning,
            ) = await memgpt_agent.step(user_message, first_message=False, skip_verify=True)

            non_user_messages = [new_message for new_message in new_messages if new_message["role"] != "user"]

            mapped_messages = map_messages_to_parsed_function_call_arguments(non_user_messages)

            print(mapped_messages)

            await websocket.send_json(
                {
                    "new_messages": mapped_messages,
                    "time": datetime.datetime.now().isoformat(),
                }
            )

    app.mount(
        "/",
        SPAStaticFiles(
            directory=os.path.join(getcwd(), "memgpt", "web", "static_files"),
            html=True,
        ),
        name="spa-static-files",
    )
