import asyncio
import json
import threading

import websockets

from memgpt.server.websocket_interface import SyncWebSocketInterface
from memgpt.server.constants import DEFAULT_PORT


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

    async def handle_client(self, websocket, path):
        self.interface.register_client(websocket)
        try:
            async for message in websocket:
                # Assuming the message is a JSON string
                data = json.loads(message)

                if data["type"] == "initialize":
                    # Handle agent initialization
                    self.agent = self.initialize_agent(data["config"])
                    await websocket.send(json.dumps({"status": "Agent initialized"}))
                elif data["type"] == "message":
                    # Handle regular agent messages
                    response = self.agent.step(data["content"])
                    await websocket.send(json.dumps({"response": response}))
                # ... handle other message types as needed ...

        except websockets.exceptions.ConnectionClosed:
            pass
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
