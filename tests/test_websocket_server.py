import asyncio
import json

import websockets
import pytest

from memgpt.server.constants import WS_DEFAULT_PORT
from memgpt.server.ws_api.server import WebSocketServer
from memgpt.config import AgentConfig


@pytest.mark.asyncio
async def test_dummy():
    assert True


@pytest.mark.skip(reason="websockets is temporarily unsupported in 0.2.12")
@pytest.mark.asyncio
async def test_websocket_server():
    # host = "127.0.0.1"
    host = "localhost"
    server = WebSocketServer(host=host)
    server_task = asyncio.create_task(server.run())  # Create a task for the server

    # the agent config we want to ask the server to instantiate with
    # test_config = AgentConfig(
    #     persona="sam_pov",
    #     human="cs_phd",
    #     preset="memgpt_chat",
    #     model_endpoint=
    # )
    test_config = {}

    uri = f"ws://{host}:{WS_DEFAULT_PORT}"
    try:
        async with websockets.connect(uri) as websocket:
            # Initialize the server with a test config
            print("Sending config to server...")
            await websocket.send(json.dumps({"type": "initialize", "config": test_config}, ensure_ascii=False))
            # Wait for the response
            response = await websocket.recv()
            print(f"Response from the agent: {response}")

            await asyncio.sleep(1)  # just in case

            # Send a message to the agent
            print("Sending message to server...")
            await websocket.send(json.dumps({"type": "message", "content": "Hello, Agent!"}, ensure_ascii=False))
            # Wait for the response
            # NOTE: we should be waiting for multiple responses
            response = await websocket.recv()
            print(f"Response from the agent: {response}")
    except (OSError, ConnectionRefusedError) as e:
        print(f"Was unable to connect: {e}")
    finally:
        server_task.cancel()  # Cancel the server task after the test
