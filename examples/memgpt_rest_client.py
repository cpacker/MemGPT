import json

from memgpt import Admin, create_client
from memgpt.memory import ChatMemory

"""
Make sure you run the MemGPT server before running this example.
```
export MEMGPT_SERVER_PASS=your_token
memgpt server
```
"""


def main():
    # Create an admin client
    admin = Admin(base_url="http://localhost:8283", token="your_token")

    # Create a user + token
    create_user_response = admin.create_user()
    user_id = create_user_response.user_id
    token = create_user_response.api_key
    print(f"Created user: {user_id} with token: {token}")

    # List available keys
    get_keys_response = admin.get_keys(user_id=user_id)
    print(f"User {user_id} has keys: {get_keys_response}")

    # Connect to the server as a user
    client = create_client(base_url="http://localhost:8283", token=token)

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

    # Delete user
    admin.delete_user(user_id=user_id)
    print(f"Deleted user: {user_id} with token: {token}")


if __name__ == "__main__":
    main()
