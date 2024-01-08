import os
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.server.ws_api.interface import SyncWebSocketInterface
import memgpt.presets.presets as presets
import memgpt.utils as utils
import memgpt.system as system
from memgpt.persistence_manager import LocalStateManager


# def test_websockets():
#     # Create the websocket interface
#     ws_interface = WebSocketInterface()

#     # Create a dummy persistence manager
#     persistence_manager = InMemoryStateManager()

#     # Create an agent and hook it up to the WebSocket interface
#     memgpt_agent = presets.create_agent_from_preset(
#         presets.DEFAULT_PRESET,
#         None,  # no agent config to provide
#         "gpt-4-1106-preview",
#         personas.get_persona_text("sam_pov"),
#         humans.get_human_text("chad"),
#         ws_interface,
#         persistence_manager,
#     )

#     user_message = system.package_user_message("Hello, is anyone there?")

#     # This should trigger calls to interface user_message and others
#     memgpt_agent.step(user_message=user_message)

#     # This should trigger the web socket to send over a
#     ws_interface.print_messages(memgpt_agent.messages)


@pytest.mark.asyncio
async def test_dummy():
    assert True


@pytest.mark.skip(reason="websockets is temporarily unsupported in 0.2.12")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Missing PG URI and/or OpenAI API key")
@pytest.mark.asyncio
async def test_websockets():
    # Mock a WebSocket connection
    mock_websocket = AsyncMock()
    # mock_websocket = Mock()

    # Create the WebSocket interface with the mocked WebSocket
    ws_interface = SyncWebSocketInterface()

    # Register the mock websocket as a client
    ws_interface.register_client(mock_websocket)

    # Create an agent and hook it up to the WebSocket interface
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        ws_interface.close()
        return
    config = MemGPTConfig.load()
    if config.openai_key is None:
        config.openai_key = api_key
        config.save()

    # Mock the persistence manager
    # create agents with defaults
    agent_config = AgentConfig(
        persona="sam_pov",
        human="basic",
        model="gpt-4-1106-preview",
        model_endpoint_type="openai",
        model_endpoint="https://api.openai.com/v1",
    )
    persistence_manager = LocalStateManager(agent_config=agent_config)

    memgpt_agent = presets.create_agent_from_preset(
        agent_config.preset,
        agent_config,
        agent_config.model,
        agent_config.persona,  # note: extracting the raw text, not pulling from a file
        agent_config.human,  # note: extracting raw text, not pulling from a file
        ws_interface,
        persistence_manager,
    )

    # Mock the user message packaging
    user_message = system.package_user_message("Hello, is anyone there?")

    # Mock the agent's step method
    # agent_step = AsyncMock()
    # memgpt_agent.step = agent_step

    # Call the step method, which should trigger interface methods
    ret = memgpt_agent.step(user_message=user_message, first_message=True, skip_verify=True)
    print("ret\n")
    print(ret)

    # Print what the WebSocket received
    print("client\n")
    for call in mock_websocket.send.mock_calls:
        # print(call)
        _, args, kwargs = call
        # args will be a tuple of positional arguments sent to the send method
        # kwargs will be a dictionary of keyword arguments sent to the send method
        print(f"Sent data: {args[0] if args else None}")
        # If you're using keyword arguments, you can print them out as well:
        # print(f"Sent data with kwargs: {kwargs}")

    # This is required for the Sync wrapper version
    ws_interface.close()

    # Assertions to ensure the step method was called
    # agent_step.assert_called_once()

    # Assertions to ensure the WebSocket interface methods are called
    # You would need to implement the logic to verify that methods like ws_interface.user_message are called
    # This will require you to have some mechanism within your WebSocketInterface to track these calls


# await test_websockets()
