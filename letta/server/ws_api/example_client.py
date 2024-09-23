import asyncio

import websockets

import letta.server.ws_api.protocol as protocol
from letta.server.constants import WS_CLIENT_TIMEOUT, WS_DEFAULT_PORT
from letta.server.utils import condition_to_stop_receiving, print_server_response

# CLEAN_RESPONSES = False  # print the raw server responses (JSON)
CLEAN_RESPONSES = True  # make the server responses cleaner

# LOAD_AGENT = None  # create a brand new agent
AGENT_NAME = "agent_26"  # load an existing agent
NEW_AGENT = False

RECONNECT_DELAY = 1
RECONNECT_MAX_TRIES = 5


async def send_message_and_print_replies(websocket, user_message, agent_id):
    """Send a message over websocket protocol and wait for the reply stream to end"""
    # Send a message to the agent
    await websocket.send(protocol.client_user_message(msg=str(user_message), agent_id=agent_id))

    # Wait for messages in a loop, since the server may send a few
    while True:
        response = await asyncio.wait_for(websocket.recv(), WS_CLIENT_TIMEOUT)
        response = json_loads(response)

        if CLEAN_RESPONSES:
            print_server_response(response)
        else:
            print(f"Server response:\n{json_dumps(response, indent=2)}")

        # Check for a specific condition to break the loop
        if condition_to_stop_receiving(response):
            break


async def basic_cli_client():
    """Basic example of a Letta CLI client that connects to a Letta server.py process via WebSockets

    Meant to illustrate how to use the server.py process, so limited in features (only supports sending user messages)
    """
    uri = f"ws://localhost:{WS_DEFAULT_PORT}"

    closed_on_message = False
    retry_attempts = 0
    while True:  # Outer loop for reconnection attempts
        try:
            async with websockets.connect(uri) as websocket:
                if NEW_AGENT:
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
                    response = json_loads(response)
                    print(f"Server response:\n{json_dumps(response, indent=2)}")

                    await asyncio.sleep(1)

                while True:
                    if closed_on_message:
                        # If we're on a retry after a disconnect, don't ask for input again
                        closed_on_message = False
                    else:
                        user_input = input("\nEnter your message: ")
                        print("\n")

                    # Send a message to the agent
                    try:
                        await send_message_and_print_replies(websocket=websocket, user_message=user_input, agent_id=AGENT_NAME)
                        retry_attempts = 0
                    except websockets.exceptions.ConnectionClosedError:
                        print("Connection to server was lost. Attempting to reconnect...")
                        closed_on_message = True
                        raise

        except websockets.exceptions.ConnectionClosedError:
            # Decide whether or not to retry the connection
            if retry_attempts < RECONNECT_MAX_TRIES:
                retry_attempts += 1
                await asyncio.sleep(RECONNECT_DELAY)  # Wait for N seconds before reconnecting
                continue
            else:
                print(f"Max attempts exceeded ({retry_attempts} > {RECONNECT_MAX_TRIES})")
                break

        except asyncio.TimeoutError:
            print("Timeout waiting for the server response.")
            continue

        except Exception as e:
            print(f"An error occurred: {e}")
            continue


asyncio.run(basic_cli_client())
