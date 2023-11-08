import os
import subprocess
import sys

from memgpt.config import MemGPTConfig, AgentConfig
from memgpt.server.websocket_interface import WebSocketInterface

import memgpt.presets as presets
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import InMemoryStateManager

import argparse


def test_websockets():
    # Create the websocket interface
    ws_interface = WebSocketInterface()

    # Create a dummy persistence manager
    persistence_manager = InMemoryStateManager()

    # Create an agent and hook it up to the WebSocket interface
    memgpt_agent = presets.use_preset(
        presets.DEFAULT_PRESET,
        None,  # no agent config to provide
        "gpt-4-1106-preview",
        personas.get_persona_text("sam_pov"),
        humans.get_human_text("chad"),
        ws_interface,
        persistence_manager,
    )
    ws_interface.print_messages(memgpt_agent.messages)


test_websockets()
