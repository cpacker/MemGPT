import json
import uuid

from letta import create_client
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory

"""
This example show how you can add LangChain tools .

First, make sure you have LangChain and some of the extras downloaded.
For this specific example, you will need `wikipedia` installed.
```
poetry install --extras "external-tools"
```
then setup letta with `letta configure`.
"""


def main():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    langchain_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    # Create a `LocalClient` (you can also use a `RESTClient`, see the letta_rest_client.py example)
    client = create_client()
    client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))

    # create tool
    # Note the additional_imports_module_attr_map
    # We need to pass in a map of all the additional imports necessary to run this tool
    # Because an object of type WikipediaAPIWrapper is passed into WikipediaQueryRun to initialize langchain_tool,
    # We need to also import WikipediaAPIWrapper
    # The map is a mapping of the module name to the attribute name
    # langchain_community.utilities.WikipediaAPIWrapper
    wikipedia_query_tool = client.load_langchain_tool(
        langchain_tool, additional_imports_module_attr_map={"langchain_community.utilities": "WikipediaAPIWrapper"}
    )
    tool_name = wikipedia_query_tool.name

    # Confirm that the tool is in
    tools = client.list_tools()
    assert wikipedia_query_tool.name in [t.name for t in tools]

    # Generate uuid for agent name for this example
    namespace = uuid.NAMESPACE_DNS
    agent_uuid = str(uuid.uuid5(namespace, "letta-langchain-tooling-example"))

    # Clear all agents
    for agent_state in client.list_agents():
        if agent_state.name == agent_uuid:
            client.delete_agent(agent_id=agent_state.id)
            print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")

    # google search persona
    persona = f"""

    My name is Letta.

    I am a personal assistant who answers a user's questions using wikipedia searches. When a user asks me a question, I will use a tool called {tool_name} which will search Wikipedia and return a Wikipedia page about the topic. It is my job to construct the best query to input into {tool_name} based on the user's question.

    Don’t forget - inner monologue / inner thoughts should always be different than the contents of send_message! send_message is how you communicate with the user, whereas inner thoughts are your own personal inner thoughts.
    """

    # Create an agent
    agent_state = client.create_agent(name=agent_uuid, memory=ChatMemory(human="My name is Matt.", persona=persona), tools=[tool_name])
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")

    # Send a message to the agent
    send_message_response = client.user_message(agent_id=agent_state.id, message="How do you pronounce Albert Einstein's name?")
    for message in send_message_response.messages:
        response_json = json.dumps(message.model_dump(), indent=4)
        print(f"{response_json}\n")

    # Delete agent
    client.delete_agent(agent_id=agent_state.id)
    print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


if __name__ == "__main__":
    main()
