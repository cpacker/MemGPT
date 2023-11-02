import os
from os import getcwd

from memgpt.agent import AgentAsync
from memgpt.config import MemGPTConfig
from memgpt.utils import parse_json


def find_last_bot_message(all_messages: list):
    for i in reversed(range(len(all_messages))):
        if all_messages[i]["role"] == "assistant" and all_messages[i]["function_call"]["name"] == "send_message":
            return all_messages[i]


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
            bot_response = find_last_bot_message(new_messages)
            bot_function_call_string_arguments = bot_response["function_call"]["arguments"]
            bot_function_call_json_arguments = parse_json(bot_function_call_string_arguments)

            await websocket.send_text(bot_function_call_json_arguments["message"])

    app.mount(
        "/",
        SPAStaticFiles(
            directory=os.path.join(getcwd(), "memgpt", "web", "static_files"),
            html=True,
        ),
        name="spa-static-files",
    )
