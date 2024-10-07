import json

from letta.schemas.agent_config import AgentConfig, AgentType
from letta.client.client import create_client

from IPython import embed


def parse_messages(messages):
    return_messages = []
    for msg in messages:
        if msg.message_type == "function_call" and msg.function_call.name == "send_message":
            arguments = json.loads(msg.function_call.arguments)
            return_messages.append(arguments["message"])
    return return_messages


def print_messages(messages):
    for n, i in enumerate(parse_messages(messages)):
        print(f"\033[94m[{n}]\033[0m {i}")


def test_split_thread_creation():
    client = create_client()
    assert client is not None

    agent_state = client.create_agent(agent_config=AgentConfig(agent_type=AgentType.split_thread_agent))
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(agent_id=agent_state.id, message="My name is Vivek.")
    assert response is not None

    print_messages(response.messages)


if __name__ == "__main__":
    test_split_thread_creation()
