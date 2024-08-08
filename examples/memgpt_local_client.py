import json

from memgpt import create_client
from memgpt.memory import ChatMemory


def main():

    # Create a `LocalClient`
    client = create_client()

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
