import json

from letta.client.client import create_client
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig


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


def print_functions(messages):
    for n, i in enumerate(messages):
        if i.message_type == "function_call":
            print(f"\033[94m[{n}]\033[0m {i.function_call.name} {i.function_call.arguments}")


def test_split_thread_creation():
    client = create_client()
    assert client is not None

    agent_state = client.create_agent(
        agent_type=AgentType.split_thread_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
    )
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(
        agent_id=agent_state.id, message="My name is Vivek. Before you respond, make sure you wait for the memory to be updated"
    )
    assert response is not None

    print_messages(response.messages)


def interact_with_agent():
    client = create_client()
    assert client is not None

    agent_state = client.create_agent(
        agent_type=AgentType.split_thread_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config(provider="openai"),
    )
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    while True:
        message = input("You: ")
        response = client.user_message(agent_id=agent_state.id, message=message)
        print_functions(response.messages)


if __name__ == "__main__":
    test_split_thread_creation()
    interact_with_agent()
