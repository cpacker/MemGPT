import asyncio
import json

import websockets
import questionary

from memgpt.server.websocket_server import WebSocketServer
from memgpt.server.constants import DEFAULT_PORT, SERVER_STEP_STOP_MESSAGE, CLIENT_TIMEOUT


def condition_to_stop_receiving(response):
    """Determines when to stop listening to the server"""
    # response_data = json.loads(response)
    # return response_data.get("type") == SERVER_STEP_STOP_MESSAGE
    return response.get("type") == "server_message" and response.get("message") == SERVER_STEP_STOP_MESSAGE


async def basic_cli_client():
    uri = f"ws://localhost:{DEFAULT_PORT}"

    async with websockets.connect(uri) as websocket:
        # Initialize agent
        print("Sending config to server...")
        await websocket.send(json.dumps({"type": "initialize", "config": {}}))
        # Wait for the response
        response = await websocket.recv()
        response = json.loads(response)
        print(f"Server response:\n{json.dumps(response, indent=2)}")

        await asyncio.sleep(1)

        while True:
            user_input = input("\n\nEnter your message: ")
            print("\n\n")

            # Send a message to the agent
            print("### Sending message to server...")
            await websocket.send(json.dumps({"type": "user_message", "content": str(user_input)}))

            # Wait for messages in a loop, since the server may send a few
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), CLIENT_TIMEOUT)
                    response = json.loads(response)
                    # print(f"Response from the server:\n{response}")
                    print(f"Server response:\n{json.dumps(response, indent=2)}")

                    # Check for a specific condition to break the loop
                    if condition_to_stop_receiving(response):
                        break
                except asyncio.TimeoutError:
                    print("Timeout waiting for the server response.")
                    break
                except websockets.exceptions.ConnectionClosedError:
                    print("Connection to server was lost.")
                    break
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

            # # Wait for the response
            # response = await websocket.recv()
            # print(f"Response from the agent: {response}")

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


asyncio.run(basic_cli_client())
