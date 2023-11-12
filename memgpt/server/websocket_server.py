import asyncio
import json
import threading

import websockets

from memgpt.server.websocket_interface import SyncWebSocketInterface
from memgpt.server.constants import DEFAULT_PORT, SERVER_STEP_START_MESSAGE, SERVER_STEP_STOP_MESSAGE
import memgpt.system as system
import memgpt.constants as memgpt_constants


def create_dummy_agent(ws_interface):
    from memgpt.config import MemGPTConfig, AgentConfig
    import memgpt.presets as presets
    import memgpt.personas.personas as personas
    import memgpt.humans.humans as humans

    # import memgpt.system as system
    from memgpt.persistence_manager import InMemoryStateManager

    # Create the WebSocket interface with the mocked WebSocket
    # ws_interface = SyncWebSocketInterface()

    # Register the mock websocket as a client
    # ws_interface.register_client(mock_websocket)

    # Mock the persistence manager
    persistence_manager = InMemoryStateManager()

    # Create an agent and hook it up to the WebSocket interface
    config = MemGPTConfig()
    memgpt_agent = presets.use_preset(
        presets.DEFAULT_PRESET,
        config,  # no agent config to provide
        "gpt-4-1106-preview",
        personas.get_persona_text("sam_pov"),
        humans.get_human_text("basic"),
        ws_interface,
        persistence_manager,
    )

    return memgpt_agent


class WebSocketServer:
    def __init__(self, host="localhost", port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.interface = SyncWebSocketInterface()
        self.agent = None

    def run_step(self, user_message, first_message=False, no_verify=False):
        while True:
            new_messages, heartbeat_request, function_failed, token_warning = self.agent.step(
                user_message, first_message=first_message, skip_verify=no_verify
            )

            if token_warning:
                user_message = system.get_token_limit_warning()
            elif function_failed:
                user_message = system.get_heartbeat(memgpt_constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
            elif heartbeat_request:
                user_message = system.get_heartbeat(memgpt_constants.REQ_HEARTBEAT_MESSAGE)
            else:
                # return control
                break

    async def handle_client(self, websocket, path):
        self.interface.register_client(websocket)
        try:
            # async for message in websocket:
            while True:
                message = await websocket.recv()

                # Assuming the message is a JSON string
                data = json.loads(message)

                if data["type"] == "initialize":
                    # Handle agent initialization
                    self.agent = self.initialize_agent(data["config"])
                    await websocket.send(json.dumps({"type": "server_message", "message": "Agent initialized"}))
                elif data["type"] == "user_message":
                    if self.agent is None:
                        await websocket.send(json.dumps({"type": "server_message", "message": "Error: no agent has been initialized"}))
                    # response = self.agent.step(data["content"])
                    # Handle regular agent messages
                    user_message = data["content"]
                    await websocket.send(json.dumps({"type": "server_message", "message": SERVER_STEP_START_MESSAGE}))
                    self.run_step(user_message)
                    await asyncio.sleep(1)  # pause before sending the terminating message, w/o this messages may be missed
                    await websocket.send(json.dumps({"type": "server_message", "message": SERVER_STEP_STOP_MESSAGE}))
                # ... handle other message types as needed ...
                else:
                    print(f"[server] unrecognized client package data type: {data}")

        except websockets.exceptions.ConnectionClosed:
            print(f"[server] connection with client was closed")
        finally:
            self.interface.unregister_client(websocket)

    def initialize_agent(self, config):
        # Initialize the agent based on the provided configuration
        print("Creating dummy agent...")
        agent = create_dummy_agent(self.interface)
        print("Created dummy agent")
        return agent

    def initialize_server(self):
        print("Server is initializing...")
        print(f"Listening on {self.host}:{self.port}...")

    async def start_server(self):
        self.initialize_server()
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

    # def run(self):
    #     asyncio.run(self.start_server())

    def run(self):
        return self.start_server()  # Return the coroutine


if __name__ == "__main__":
    server = WebSocketServer()
    # server.run()
    asyncio.run(server.run())
