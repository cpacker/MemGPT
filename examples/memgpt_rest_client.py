import json

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

    # Create an agent
    agent_state = client.create_agent(name="my_agent", memory=ChatMemory(human="My name is Sarah.", persona="I am a friendly AI."))
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")

    # Send a message to the agent
    print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")
    send_message_response = client.user_message(agent_id=agent_state.id, message="Whats my name?")
    print(f"Recieved response: \n{json.dumps(send_message_response.messages, indent=4)}")

    # Delete agent
    client.delete_agent(agent_id=agent_state.id)
    print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


if __name__ == "__main__":
    main()
