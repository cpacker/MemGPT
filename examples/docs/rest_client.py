from letta import create_client
from letta.schemas.memory import ChatMemory

"""
Make sure you run the Letta server before running this example.
```
letta server
```
"""


def main():
    # Connect to the server as a user
    client = create_client(base_url="http://localhost:8283")

    # list available configs on the server
    llm_configs = client.list_llm_configs()
    print(f"Available LLM configs: {llm_configs}")
    embedding_configs = client.list_embedding_configs()
    print(f"Available embedding configs: {embedding_configs}")

    # Create an agent
    agent_state = client.create_agent(
        name="my_agent",
        memory=ChatMemory(human="My name is Sarah.", persona="I am a friendly AI."),
        embedding_config=embedding_configs[0],
        llm_config=llm_configs[0],
    )
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")

    # Send a message to the agent
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")
    response = client.user_message(agent_id=agent_state.id, message="Whats my name?")
    print(f"Received response:", response.messages)

    # Delete agent
    client.delete_agent(agent_id=agent_state.id)
    print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


if __name__ == "__main__":
    main()
