import json
import uuid

from letta import create_client
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory

"""
This example show how you can add CrewAI tools .

First, make sure you have CrewAI and some of the extras downloaded.
```
poetry install --extras "external-tools"
```
then setup letta with `letta configure`.
"""


def main():
    from crewai_tools import ScrapeWebsiteTool

    crewai_tool = ScrapeWebsiteTool(website_url="https://www.example.com")

    # Create a `LocalClient` (you can also use a `RESTClient`, see the letta_rest_client.py example)
    client = create_client()
    client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))

    # create tool
    example_website_scrape_tool = client.load_crewai_tool(crewai_tool)
    tool_name = example_website_scrape_tool.name

    # Confirm that the tool is in
    tools = client.list_tools()
    assert example_website_scrape_tool.name in [t.name for t in tools]

    # Generate uuid for agent name for this example
    namespace = uuid.NAMESPACE_DNS
    agent_uuid = str(uuid.uuid5(namespace, "letta-crewai-tooling-example"))

    # Clear all agents
    for agent_state in client.list_agents():
        if agent_state.name == agent_uuid:
            client.delete_agent(agent_id=agent_state.id)
            print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")

    # google search persona
    persona = f"""

    My name is Letta.

    I am a personal assistant who answers a user's questions about a website `example.com`. When a user asks me a question about `example.com`, I will use a tool called {tool_name} which will search `example.com` and answer the relevant question.

    Don’t forget - inner monologue / inner thoughts should always be different than the contents of send_message! send_message is how you communicate with the user, whereas inner thoughts are your own personal inner thoughts.
    """

    # Create an agent
    agent_state = client.create_agent(name=agent_uuid, memory=ChatMemory(human="My name is Matt.", persona=persona), tools=[tool_name])
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")

    # Send a message to the agent
    send_message_response = client.user_message(agent_id=agent_state.id, message="What's on the example.com website?")
    for message in send_message_response.messages:
        response_json = json.dumps(message.model_dump(), indent=4)
        print(f"{response_json}\n")

    # Delete agent
    client.delete_agent(agent_id=agent_state.id)
    print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


if __name__ == "__main__":
    main()
