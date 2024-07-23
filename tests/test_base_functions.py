import os

import pytest

import memgpt.functions.function_sets.base as base_functions
from memgpt import create_client

from .utils import create_config, wipe_config

# test_agent_id = "test_agent"
client = None


@pytest.fixture(scope="module")
def agent_obj():
    """Create a test agent that we can call functions on"""
    wipe_config()
    global client
    if os.getenv("OPENAI_API_KEY"):
        create_config("openai")
    else:
        create_config("memgpt_hosted")

    client = create_client()

    agent_state = client.create_agent()

    global agent_obj
    agent_obj = client.server._get_or_load_agent(user_id=client.user_id, agent_id=agent_state.id)
    yield agent_obj

    client.delete_agent(agent_obj.agent_state.id)


def test_archival(agent_obj):
    base_functions.archival_memory_insert(agent_obj, "banana")
    base_functions.archival_memory_search(agent_obj, "banana")
    base_functions.archival_memory_search(agent_obj, "banana", page=0)


def test_recall(agent_obj):
    base_functions.conversation_search(agent_obj, "banana")
    base_functions.conversation_search(agent_obj, "banana", page=0)
    base_functions.conversation_search_date(agent_obj, start_date="2022-01-01", end_date="2022-01-02")
