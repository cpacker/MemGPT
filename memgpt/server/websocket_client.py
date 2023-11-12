import asyncio
import json

import websockets

from memgpt.server.websocket_server import WebSocketServer
from memgpt.server.constants import DEFAULT_PORT


async def test_agent():
    uri = f"ws://localhost:{DEFAULT_PORT}"

    async with websockets.connect(uri) as websocket:
        # Initialize agent
        print("Sending config to server...")
        await websocket.send(json.dumps({"type": "initialize", "config": {}}))
        # Wait for the response
        response = await websocket.recv()
        print(f"Response from the agent: {response}")

        await asyncio.sleep(1)

        # Send a message to the agent
        print("Sending message to server...")
        await websocket.send(json.dumps({"type": "message", "content": "Hello, Agent!"}))
        # Wait for the response
        response = await websocket.recv()
        print(f"Response from the agent: {response}")

        # # Send a message to the server
        # await websocket.send("Hello, Agent!")

        # # Wait for the response
        # response = await websocket.recv()
        # print(f"Response from the agent: {response}")

        # # Send another message
        # await websocket.send("Another message")

        # # Wait for the response
        # response = await websocket.recv()
        # print(f"Response from the agent: {response}")


asyncio.run(test_agent())
