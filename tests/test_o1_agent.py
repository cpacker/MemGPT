import json

from letta.client.client import create_client
from letta.constants import DEFAULT_HUMAN
from letta.o1_agent import send_final_message, send_thinking_message
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.memory import ChatMemory
from letta.utils import get_human_text, get_persona_text


def parse_messages(messages):
    return_messages = []
    for msg in messages:
        import pdb

        pdb.set_trace()
        arguments = json.loads(msg.function_call.arguments)
        return_messages.append(arguments["message"])
    return return_messages


def print_messages(messages):
    for n, i in enumerate(parse_messages(messages)):
        print(f"\033[94m[{n}]\033[0m {i}")


def test_o1_agent():
    client = create_client()
    assert client is not None

    thinking_tool = client.create_tool(send_thinking_message)
    final_tool = client.create_tool(send_final_message)

    agent_state = client.create_agent(
        agent_type=AgentType.o1_agent,
        tools=[thinking_tool.name, final_tool.name],
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        memory=ChatMemory(human=get_human_text(DEFAULT_HUMAN), persona=get_persona_text("o1_persona")),
    )
    agent = client.get_agent(agent_id=agent_state.id)
    assert agent is not None

    response = client.user_message(agent_id=agent_state.id, message="How many Rs are there in strawberry?")
    assert response is not None
    assert len(response.messages) > 3
    print("\n\n".join([str(i) for i in response.messages]))
    # print_messages(response.messages)


if __name__ == "__main__":
    test_o1_agent()
