import asyncio
import json

import websockets

import memgpt.server.websocket_protocol as protocol
from memgpt.server.websocket_server import WebSocketServer
from memgpt.server.constants import DEFAULT_PORT, SERVER_STEP_STOP_MESSAGE, CLIENT_TIMEOUT


# CLEAN_RESPONSES = False  # print the raw server responses (JSON)
CLEAN_RESPONSES = True  # make the server responses cleaner


def condition_to_stop_receiving(response):
    """Determines when to stop listening to the server"""
    return response.get("type") == "agent_response_end"


def print_server_response(response):
    """Turn response json into a nice print"""
    if response["type"] == "agent_response_start":
        print("[agent.step start]")
    elif response["type"] == "agent_response_end":
        print("[agent.step end]")
    elif response["type"] == "agent_response":
        msg = response["message"]
        if response["message_type"] == "internal_monologue":
            print(f"[inner thoughts] {msg}")
        elif response["message_type"] == "assistant_message":
            print(f"{msg}")
        elif response["message_type"] == "function_message":
            pass
        else:
            print(response)
    else:
        print(response)


async def basic_cli_client():
    """Basic example of a MemGPT CLI client that connects to a MemGPT server.py process via WebSockets

    Meant to illustrate how to use the server.py process, so limited in features (only supports sending user messages)
    """
    uri = f"ws://localhost:{DEFAULT_PORT}"

    async with websockets.connect(uri) as websocket:
        # Initialize agent
        print("Sending config to server...")
        # await websocket.send(json.dumps({"type": "initialize", "config": {}}))
        example_config = {}
        await websocket.send(protocol.client_command_init(example_config))
        # Wait for the response
        response = await websocket.recv()
        response = json.loads(response)
        print(f"Server response:\n{json.dumps(response, indent=2)}")

        await asyncio.sleep(1)

        while True:
            user_input = input("\nEnter your message: ")
            print("\n")

            # Send a message to the agent
            await websocket.send(protocol.client_user_message(str(user_input)))

            # Wait for messages in a loop, since the server may send a few
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), CLIENT_TIMEOUT)
                    response = json.loads(response)

                    if CLEAN_RESPONSES:
                        print_server_response(response)
                    else:
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


asyncio.run(basic_cli_client())
