from typing import Union

from letta import LocalClient, RESTClient


def cleanup(client: Union[LocalClient, RESTClient], agent_uuid: str):
    # Clear all agents
    for agent_state in client.list_agents():
        if agent_state.name == agent_uuid:
            client.delete_agent(agent_id=agent_state.id)
            print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")
