import asyncio
import json

import websockets

import memgpt.server.websocket_protocol as protocol
from memgpt.server.websocket_server import WebSocketServer
from memgpt.server.constants import DEFAULT_PORT, CLIENT_TIMEOUT
from memgpt.server.utils import condition_to_stop_receiving, print_server_response


# CLEAN_RESPONSES = False  # print the raw server responses (JSON)
CLEAN_RESPONSES = True  # make the server responses cleaner

# LOAD_AGENT = None  # create a brand new agent
LOAD_AGENT = "agent_26"  # load an existing agent


async def basic_cli_client():
    """Basic example of a MemGPT CLI client that connects to a MemGPT server.py process via WebSockets

    Meant to illustrate how to use the server.py process, so limited in features (only supports sending user messages)
    """
    uri = f"ws://localhost:{DEFAULT_PORT}"

    async with websockets.connect(uri) as websocket:
        if LOAD_AGENT is not None:
            # Load existing agent
            print("Sending load message to server...")
            await websocket.send(protocol.client_command_load(LOAD_AGENT))

        else:
            # Initialize new agent
            print("Sending config to server...")
            example_config = {
                "persona": "sam_pov",
                "human": "cs_phd",
                "model": "gpt-4-1106-preview",  # gpt-4-turbo
            }
            await websocket.send(protocol.client_command_create(example_config))
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
