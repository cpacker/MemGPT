from typing import Union

from letta import LocalClient, RESTClient
from letta.functions.functions import parse_source_code
from letta.functions.schema_generator import generate_schema
from letta.schemas.tool import Tool


def cleanup(client: Union[LocalClient, RESTClient], agent_uuid: str):
    # Clear all agents
    for agent_state in client.list_agents():
        if agent_state.name == agent_uuid:
            client.delete_agent(agent_id=agent_state.id)
            print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


# Utility functions
def create_tool_from_func(func: callable):
    return Tool(
        name=func.__name__,
        description="",
        source_type="python",
        tags=[],
        source_code=parse_source_code(func),
        json_schema=generate_schema(func, None),
    )
