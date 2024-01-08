from memgpt import MemGPT
from memgpt import constants
import memgpt.functions.function_sets.base as base_functions

from .utils import wipe_config


# test_agent_id = "test_agent"
client = None
agent_obj = None


def create_test_agent():
    """Create a test agent that we can call functions on"""
    wipe_config()
    global client
    client = MemGPT(quickstart="openai")

    agent_state = client.create_agent(
        agent_config={
            # "name": test_agent_id,
            "persona": constants.DEFAULT_PERSONA,
            "human": constants.DEFAULT_HUMAN,
        }
    )

    global agent_obj
    agent_obj = client.server._get_or_load_agent(user_id="NULL", agent_id=agent_state.id)


def test_archival():
    global agent_obj
    if agent_obj is None:
        create_test_agent()
    assert agent_obj is not None

    base_functions.archival_memory_insert(agent_obj, "banana")

    base_functions.archival_memory_search(agent_obj, "banana")
    base_functions.archival_memory_search(agent_obj, "banana", page=0)


def test_recall():
    global agent_obj
    if agent_obj is None:
        create_test_agent()

    base_functions.conversation_search(agent_obj, "banana")
    base_functions.conversation_search(agent_obj, "banana", page=0)

    base_functions.conversation_search_date(agent_obj, start_date="2022-01-01", end_date="2022-01-02")
