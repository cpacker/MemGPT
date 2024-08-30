from memgpt.client.client import create_client


def parse_messages(messages):
    return [i.text for i in messages]
    parsed_messages = {}
    for message in messages:
        parsed_messages[message.id] = message
    sorted_messages = sorted(parsed_messages.values(), key=lambda x: x.created_at)
    return [i.text for i in sorted_messages]


def print_messages(messages):
    for n, i in enumerate(parse_messages(messages)):
        print(f"\033[94m[{n}]\033[0m {i}")


def test_split_thread_creation():
    client = create_client()
    assert client is not None

    agent_state = client.create_agent(split_thread_agent=True)
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(agent_id=agent_state.id, message="My name is Vivek.")
    assert response is not None

    print_messages(response.messages)


if __name__ == "__main__":
    test_split_thread_creation()
