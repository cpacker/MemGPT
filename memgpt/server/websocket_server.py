import asyncio
import json
import traceback

import websockets

from memgpt.server.websocket_interface import SyncWebSocketInterface
from memgpt.server.constants import DEFAULT_PORT
import memgpt.server.websocket_protocol as protocol
import memgpt.system as system
import memgpt.constants as memgpt_constants


class WebSocketServer:
    def __init__(self, host="localhost", port=DEFAULT_PORT):
        self.host = host
        self.port = port
        self.interface = SyncWebSocketInterface()

        self.agent = None
        self.agent_name = None

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
                try:
                    data = json.loads(message)
                except:
                    print(f"[server] bad data from client:\n{data}")
                    await websocket.send(protocol.server_command_response(f"Error: bad data from client - {str(data)}"))
                    continue

                if "type" not in data:
                    print(f"[server] bad data from client (JSON but no type):\n{data}")
                    await websocket.send(protocol.server_command_response(f"Error: bad data from client - {str(data)}"))

                elif data["type"] == "command":
                    # Create a new agent
                    if data["command"] == "create_agent":
                        try:
                            self.agent = self.create_new_agent(data["config"])
                            await websocket.send(protocol.server_command_response("OK: Agent initialized"))
                        except Exception as e:
                            self.agent = None
                            print(f"[server] self.create_new_agent failed with:\n{e}")
                            print(f"{traceback.format_exc()}")
                            await websocket.send(protocol.server_command_response(f"Error: Failed to init agent - {str(e)}"))

                    # Load an existing agent
                    elif data["command"] == "load_agent":
                        agent_name = data.get("name")
                        if agent_name is not None:
                            try:
                                self.agent = self.load_agent(agent_name)
                                self.agent_name = agent_name
                                await websocket.send(protocol.server_command_response(f"OK: Agent '{agent_name}' loaded"))
                            except Exception as e:
                                print(f"[server] self.load_agent failed with:\n{e}")
                                print(f"{traceback.format_exc()}")
                                self.agent = None
                                await websocket.send(
                                    protocol.server_command_response(f"Error: Failed to load agent '{agent_name}' - {str(e)}")
                                )
                        else:
                            await websocket.send(protocol.server_command_response(f"Error: 'name' not provided"))

                    else:
                        print(f"[server] unrecognized client command type: {data}")
                        await websocket.send(protocol.server_error(f"unrecognized client command type: {data}"))

                elif data["type"] == "user_message":
                    user_message = data["message"]

                    if "agent_name" in data:
                        agent_name = data["agent_name"]
                        # If the agent requested the same one that's already loading?
                        if self.agent_name is None or self.agent_name != data["agent_name"]:
                            try:
                                print(f"[server] loading agent {agent_name}")
                                self.agent = self.load_agent(agent_name)
                                self.agent_name = agent_name
                                # await websocket.send(protocol.server_command_response(f"OK: Agent '{agent_name}' loaded"))
                            except Exception as e:
                                print(f"[server] self.load_agent failed with:\n{e}")
                                print(f"{traceback.format_exc()}")
                                self.agent = None
                                await websocket.send(
                                    protocol.server_command_response(f"Error: Failed to load agent '{agent_name}' - {str(e)}")
                                )
                    else:
                        await websocket.send(protocol.server_agent_response_error("agent_name was not specified in the request"))
                        continue

                    if self.agent is None:
                        await websocket.send(protocol.server_agent_response_error("No agent has been initialized"))
                    else:
                        await websocket.send(protocol.server_agent_response_start())
                        try:
                            self.run_step(user_message)
                        except Exception as e:
                            print(f"[server] self.run_step failed with:\n{e}")
                            print(f"{traceback.format_exc()}")
                            await websocket.send(protocol.server_agent_response_error(f"self.run_step failed with: {e}"))

                        await asyncio.sleep(1)  # pause before sending the terminating message, w/o this messages may be missed
                        await websocket.send(protocol.server_agent_response_end())

                # ... handle other message types as needed ...
                else:
                    print(f"[server] unrecognized client package data type: {data}")
                    await websocket.send(protocol.server_error(f"unrecognized client package data type: {data}"))

        except websockets.exceptions.ConnectionClosed:
            print(f"[server] connection with client was closed")
        finally:
            # TODO autosave the agent

            self.interface.unregister_client(websocket)

    def create_new_agent(self, config):
        """Config is json that arrived over websocket, so we need to turn it into a config object"""
        from memgpt.config import AgentConfig
        import memgpt.presets.presets as presets
        import memgpt.utils as utils
        from memgpt.persistence_manager import InMemoryStateManager

        print("Creating new agent...")

        # Initialize the agent based on the provided configuration
        agent_config = AgentConfig(**config)

        # Use an in-state persistence manager
        persistence_manager = InMemoryStateManager()

        # Create agent via preset from config
        agent = presets.use_preset(
            agent_config.preset,
            agent_config,
            agent_config.model,
            utils.get_persona_text(agent_config.persona),
            utils.get_human_text(agent_config.human),
            self.interface,
            persistence_manager,
        )
        print("Created new agent from config")

        return agent

    def load_agent(self, agent_name):
        """Load an agent from a directory"""
        import memgpt.utils as utils
        from memgpt.config import AgentConfig
        from memgpt.agent import Agent

        print(f"Loading agent {agent_name}...")

        agent_files = utils.list_agent_config_files()
        agent_names = [AgentConfig.load(f).name for f in agent_files]

        if agent_name not in agent_names:
            raise ValueError(f"agent '{agent_name}' does not exist")

        agent_config = AgentConfig.load(agent_name)
        agent = Agent.load_agent(self.interface, agent_config)
        print("Created agent by loading existing config")

        return agent

    def initialize_server(self):
        print("Server is initializing...")
        print(f"Listening on {self.host}:{self.port}...")

    async def start_server(self):
        self.initialize_server()
        async with websockets.serve(self.handle_client, self.host, self.port):
            await asyncio.Future()  # Run forever

    def run(self):
        return self.start_server()  # Return the coroutine


if __name__ == "__main__":
    server = WebSocketServer()
    asyncio.run(server.run())
