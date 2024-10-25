import json
import uuid

from letta import create_client
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory

"""
Setup here.
"""
# Create a `LocalClient` (you can also use a `RESTClient`, see the letta_rest_client.py example)
client = create_client()
client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini"))
client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))

# Generate uuid for agent name for this example
namespace = uuid.NAMESPACE_DNS
agent_uuid = str(uuid.uuid5(namespace, "letta-composio-tooling-example"))

# Clear all agents
for agent_state in client.list_agents():
    if agent_state.name == agent_uuid:
        client.delete_agent(agent_id=agent_state.id)
        print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


"""
This example show how you can add Composio tools .

First, make sure you have Composio and some of the extras downloaded.
```
poetry install --extras "external-tools"
```
then setup letta with `letta configure`.

Aditionally, this example stars a Github repo on your behalf. You will need to configure Composio in your environment.
```
composio login
composio add github
```

Last updated Oct 2, 2024. Please check `composio` documentation for any composio related issues.
"""


def main():
    from composio_langchain import Action

    # Add the composio tool
    tool = client.load_composio_tool(action=Action.GITHUB_STAR_A_REPOSITORY_FOR_THE_AUTHENTICATED_USER)

    persona = f"""
    My name is Letta.

    I am a personal assistant that helps star repos on Github. It is my job to correctly input the owner and repo to the {tool.name} tool based on the user's request.

    Don’t forget - inner monologue / inner thoughts should always be different than the contents of send_message! send_message is how you communicate with the user, whereas inner thoughts are your own personal inner thoughts.
    """

    # Create an agent
    agent = client.create_agent(name=agent_uuid, memory=ChatMemory(human="My name is Matt.", persona=persona), tools=[tool.name])
    print(f"Created agent: {agent.name} with ID {str(agent.id)}")

    # Send a message to the agent
    send_message_response = client.user_message(agent_id=agent.id, message="Star a repo composio with owner composiohq on GitHub")
    for message in send_message_response.messages:
        response_json = json.dumps(message.model_dump(), indent=4)
        print(f"{response_json}\n")

    # Delete agent
    client.delete_agent(agent_id=agent.id)
    print(f"Deleted agent: {agent.name} with ID {str(agent.id)}")


if __name__ == "__main__":
    main()
