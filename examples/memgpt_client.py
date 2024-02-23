from memgpt import create_client, Admin
from memgpt.constants import DEFAULT_PRESET, DEFAULT_HUMAN, DEFAULT_PERSONA


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
    user_id, token = admin.create_user()
    print(f"Created user: {user_id} with token: {token}")

    # Connect to the server as a user
    client = create_client(base_url="http://localhost:8283", token=token)

    # Create an agent
    agent_info = client.create_agent(name="my_agent", preset=DEFAULT_PRESET, persona=DEFAULT_PERSONA, human=DEFAULT_HUMAN)
    print(f"Created agent: {agent_info.name} with ID {str(agent_info.id)}")

    # Send a message to the agent
    messages = client.user_message(agent_id=agent_info.id, message="Hello, agent!")
    print(f"Recieved response: {messages}")

    # TODO: get agent memory

    # TODO: Update agent persona

    # Delete agent
    client.delete_agent(agent_id=agent_info.id)
    print(f"Deleted agent: {agent_info.name} with ID {str(agent_info.id)}")

    # Delete user
    admin.delete_user(user_id=user_id)
    print(f"Deleted user: {user_id} with token: {token}")


if __name__ == "__main__":
    main()
