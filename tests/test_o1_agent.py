from letta.client.client import create_client
from letta.o1_agent import send_final_message, send_thinking_message


def test_o1_agent():
    client = create_client()
    assert client is not None

    thinking_tool = client.create_tool(send_thinking_message)
    final_tool = client.create_tool(send_final_message)

    agent_state = client.create_agent(tools=[thinking_tool.name, final_tool.name])
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(agent_id=agent_state.id, message="How many Rs are there in strawberry?")
    assert response is not None


if __name__ == "__main__":
    test_o1_agent()
