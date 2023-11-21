from typing import Annotated

from fastapi import Query

from memgpt import constants, system
from memgpt.server.websocket_interface import SyncWebSocketInterface
import memgpt.server.websocket_protocol as protocol


def setup_chat_ws_router():
    from fastapi import APIRouter
    from starlette.websockets import WebSocket

    router = APIRouter()

    interface = SyncWebSocketInterface()

    @router.websocket("/chat")
    async def websocket_endpoint(
        websocket: WebSocket,
        agent: Annotated[str, Query()],
    ):
        try:
            interface.register_client(websocket)
            memgpt_agent = load_agent(interface, agent)

            await websocket.accept()

            first_message = True
            skip_next_user_input = True
            internal_message = "User is back. Let's start the conversation..."
            while True:
                user_message = system.package_user_message(internal_message if skip_next_user_input else await websocket.receive_text())

                if not skip_next_user_input or first_message:
                    await websocket.send_text(protocol.server_agent_response_start())

                try:
                    (
                        new_messages,
                        user_message,
                        skip_next_user_input,
                    ) = process_agent_step(memgpt_agent, user_message, True, first_message)

                    if skip_next_user_input:
                        internal_message = user_message
                        continue

                    await websocket.send_text(protocol.server_agent_response_end())

                    if first_message:
                        first_message = False
                        internal_message = ""

                    memgpt_agent.save()
                except Exception as e:
                    print(f"[server] self.run_step failed with:\n{e}")
                    skip_next_user_input = False
                    await websocket.send_text(protocol.server_agent_response_error(f"self.run_step failed with: {e}"))
        except Exception as e:
            interface.unregister_client(websocket)

    return router


def process_agent_step(memgpt_agent, user_message, no_verify, first_message):
    new_messages, heartbeat_request, function_failed, token_warning = memgpt_agent.step(
        user_message, first_message=first_message, skip_verify=no_verify
    )

    print(new_messages[-1])

    skip_next_user_input = False
    if new_messages[-1]["role"] == "function" and new_messages[-1]["name"] != "send_message":
        last_assistant_message = next((x for x in reversed(new_messages) if x["role"] == "assistant"), None)
        user_message = (
            f"Let's continue the conversation with the user based on this internal dialog of yours: " f"{last_assistant_message['content']}"
        )
        skip_next_user_input = True
    if token_warning:
        user_message = system.get_token_limit_warning()
        skip_next_user_input = True
    elif function_failed:
        user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
        skip_next_user_input = True
    elif heartbeat_request:
        user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
        skip_next_user_input = True

    return new_messages, user_message, skip_next_user_input


def load_agent(interface, agent_name):
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
    agent = Agent.load_agent(interface, agent_config)
    print("Created agent by loading existing config")

    return agent
