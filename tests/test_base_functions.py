import pytest

import memgpt.seeds.function_sets.base as base_functions
from memgpt import create_client
from memgpt.settings import settings

client = None


@pytest.fixture(scope="module")
def agent_obj(config):
    """Create a test agent that we can call functions on"""
    global client
    client = create_client(config=config)

    agent_state = client.create_agent(
        preset=settings.preset,
    )

    global agent_obj
    agent_obj = client.server._get_or_load_agent(agent_id=agent_state.id)
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
