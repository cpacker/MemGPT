import json

from regex import W

from letta import BasicBlockMemory
from letta.client.client import Block, create_client
from letta.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from letta.offline_memory_agent import (
    finish_rethinking_memory,
    rethink_memory,
    trigger_rethink_memory,
    trigger_rethink_memory_convo,
)
from letta.prompts import gpt_system
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.message import MessageCreate
from letta.schemas.tool_rule import TerminalToolRule
from letta.utils import get_human_text, get_persona_text


def test_chat_offline_memory():
    # Check that the agent can edit multiple blocks of memory
    client = create_client()
    assert client is not None

    trigger_rethink_memory_convo_tool = client.create_tool(trigger_rethink_memory_convo)

    conversation_human_block = Block(name="chat_agent_human", label="chat_agent_human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
    conversation_persona_block = Block(
        name="chat_agent_persona", label="chat_agent_persona", value=get_persona_text(DEFAULT_PERSONA), limit=2000
    )
    offline_persona_block = Block(
        name="offline_memory_persona", label="offline_memory_persona", value=get_persona_text("offline_memory_persona"), limit=2000
    )

    conversation_human_block_new = Block(
        name="chat_agent_human_new", label="chat_agent_human_new", value=get_human_text(DEFAULT_HUMAN), limit=2000
    )
    conversation_persona_block_new = Block(
        name="chat_agent_persona_new", label="chat_agent_persona_new", value=get_persona_text(DEFAULT_PERSONA), limit=2000
    )
    conversation_messages_block = Block(name="conversation_block", label="conversation_block", value="", limit=20000)
    conversation_memory = BasicBlockMemory(blocks=[conversation_persona_block, conversation_human_block, conversation_messages_block])
    offline_memory = BasicBlockMemory(
        blocks=[
            offline_persona_block,
            conversation_human_block,
            conversation_persona_block,
            conversation_human_block_new,
            conversation_persona_block_new,
            conversation_messages_block,
        ]
    )

    conversation_agent = client.create_agent(
        name="conversation_agent",
        agent_type=AgentType.memgpt_agent,
        system=gpt_system.get_system_text("memgpt_convo_only"),
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=["send_message", trigger_rethink_memory_convo_tool.name],
        memory=conversation_memory,
        include_base_tools=False,
        include_memory_tools=False,
    )
    assert conversation_agent is not None
    assert [tool.name for tool in client.get_tools_from_agent(agent_id=conversation_agent.id)] == [
        "send_message",
        trigger_rethink_memory_convo_tool.name,
    ]

    rethink_memory_tool = client.create_tool(rethink_memory)
    finish_rethinking_memory_tool = client.create_tool(finish_rethinking_memory)
    offline_memory_agent = client.create_agent(
        name="offline_memory_agent",
        agent_type=AgentType.offline_memory_agent,
        system=gpt_system.get_system_text("memgpt_offline_memory_chat"),
        memory=offline_memory,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[rethink_memory_tool.name, finish_rethinking_memory_tool.name],
        tool_rules=[TerminalToolRule(tool_name=finish_rethinking_memory_tool.name)],
        include_base_tools=False,
    )
    assert offline_memory_agent is not None

    for message in ["Hi there", "No, my first name is Swoodily"]:
        _ = client.user_message(agent_id=conversation_agent.id, message=message)

    offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)
    _ = client.user_message(agent_id=conversation_agent.id, message="[trigger_rethink_memory]")
    offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)
    assert offline_memory_agent.memory.get_block("chat_agent_human_new").value != get_human_text(DEFAULT_HUMAN)
    conversation_agent = client.get_agent(agent_id=conversation_agent.id)
    assert (
        offline_memory_agent.memory.get_block("chat_agent_human_new").value == conversation_agent.memory.get_block("chat_agent_human").value
    )


def test_simple_colors():
    client = create_client()
    assert client is not None

    conversation_human_block = Block(name="chat_agent_human", label="chat_agent_human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
    conversation_persona_block = Block(
        name="chat_agent_persona", label="chat_agent_persona", value=get_persona_text(DEFAULT_PERSONA), limit=2000
    )
    offline_rethink_memory = Block(name="rethink_memory_block", label="rethink_memory_block", value="[empty]", limit=2000)
    conversation_memory = BasicBlockMemory(blocks=[conversation_persona_block, conversation_human_block, offline_rethink_memory])

    client = create_client()
    chat_only_agent = client.create_agent(
        name="conversation_agent",
        agent_type=AgentType.chat_only_agent,
        llm_config=LLMConfig.default_config("gpt-4o-mini"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=["send_message"],
        memory=conversation_memory,
        include_base_tools=False,
        include_memory_tools=False,
    )

    interaction = [
        {"role": "assistant", "content": "Here are two colors: Pink and Turquoise. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Turquoise."},
        {"role": "assistant", "content": "Here are two colors: Turquoise and Silver. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Silver."},
        {"role": "assistant", "content": "Here are two colors: Burgundy and Lavender. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Lavender."},
        {"role": "assistant", "content": "Here are two colors: Lavender and Amber. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Lavender."},
        {"role": "assistant", "content": "Here are two colors: Lime and Burgundy. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Lime."},
        {"role": "assistant", "content": "Here are two colors: Amber and Silver. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Amber."},
        {"role": "assistant", "content": "Here are two colors: Pink and Amber. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Amber."},
        {"role": "assistant", "content": "Here are two colors: Pink and Lavender. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Lavender."},
        {"role": "assistant", "content": "Here are two colors: Burgundy and Azure. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Burgundy."},
        {"role": "assistant", "content": "Here are two colors: Silver and Charcoal. Please pick the one you like best."},
        {"role": "user", "content": "I prefer Charcoal."},
        {"role": "user", "content": "What would I prefere between Lime and Turquoise? Answer with only the color name."},
    ]
    # messages = [MessageCreate(role=turn["role"], text=turn["content"]) for interaction in interactions for turn in interaction]
    messages = [MessageCreate(role=turn["role"], text=turn["content"]) for turn in interaction]
    assert chat_only_agent is not None
    assert chat_only_agent.memory.list_block_labels() == ["chat_agent_persona", "chat_agent_human", "rethink_memory_block"]
    response = client.send_messages(agent_id=chat_only_agent.id, messages=messages)
    chat_only_agent = client.get_agent(agent_id=chat_only_agent.id)
    print(chat_only_agent.memory)
    assert json.loads(response.messages[1].function_call.arguments)["message"] == "Lime"


def test_ripple_edit():
    client = create_client()
    assert client is not None
    print('test')

    trigger_rethink_memory_tool = client.create_tool(trigger_rethink_memory)

    conversation_human_block = Block(name="human", label="human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
    conversation_persona_block = Block(name="persona", label="persona", value=get_persona_text(DEFAULT_PERSONA), limit=2000)
    offline_human_block = Block(name="human", label="human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
    offline_persona_block = Block(name="persona", label="persona", value=get_persona_text("offline_memory_persona"), limit=2000)

    # Figure 1. from Evaluating the Ripple Effects of Knowledge Editing in Language Models (Cohen et al., 2023)
    # https://arxiv.org/pdf/2307.12976
    fact_block = Block(
        name="fact_block",
        label="fact_block",
        value="""Messi resides in the Paris.
               Messi plays in the league Ligue 1.
               Messi plays for the team Paris Saint-Germain.
               The national team Messi plays for is the Argentina team.
               Messi is also known as Leo Messi
               Victor Ulloa plays for Inter Miami""",
        limit=2000,
    )

    new_memory = Block(name="rethink_memory_block", label="rethink_memory_block", value="[empty]", limit=2000)
    conversation_memory = BasicBlockMemory(blocks=[conversation_persona_block, conversation_human_block, fact_block, new_memory])
    offline_memory = BasicBlockMemory(blocks=[offline_persona_block, offline_human_block, fact_block, new_memory])

    conversation_agent = client.create_agent(
        name="conversation_agent",
        agent_type=AgentType.memgpt_agent,
        system=gpt_system.get_system_text("memgpt_convo_only"),
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=["send_message", trigger_rethink_memory_tool.name],
        memory=conversation_memory,
        include_base_tools=False,
    )
    assert conversation_agent is not None

    assert set(conversation_agent.memory.list_block_labels()) == set([
        "persona",
        "human",
        "fact_block",
        "rethink_memory_block",
    ])

    rethink_memory_tool = client.create_tool(rethink_memory)
    finish_rethinking_memory_tool = client.create_tool(finish_rethinking_memory)
    offline_memory_agent = client.create_agent(
        name="offline_memory_agent",
        agent_type=AgentType.offline_memory_agent,
        system=gpt_system.get_system_text("memgpt_offline_memory"),
        memory=offline_memory,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[rethink_memory_tool.name, finish_rethinking_memory_tool.name],
        tool_rules=[TerminalToolRule(tool_name=finish_rethinking_memory_tool.name)],
        include_base_tools=False,
    )
    assert offline_memory_agent is not None
    assert set(offline_memory_agent.memory.list_block_labels())== set(["persona", "human", "fact_block", "rethink_memory_block"])
    _ = client.user_message(
        agent_id=conversation_agent.id, message="[trigger_rethink_memory]: Messi has now moved to playing for Inter Miami"
    )
    offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)

    assert offline_memory_agent.memory.get_block("rethink_memory_block").value != "[empty]"
    conversation_agent = client.get_agent(agent_id=conversation_agent.id)
    assert conversation_agent.memory.get_block("rethink_memory_block").value != "[empty]"


def test_chat_only_agent():
    conversation_human_block = Block(name="chat_agent_human", label="chat_agent_human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
    conversation_persona_block = Block(
        name="chat_agent_persona", label="chat_agent_persona", value=get_persona_text(DEFAULT_PERSONA), limit=2000
    )
    conversation_memory = BasicBlockMemory(blocks=[conversation_persona_block, conversation_human_block])

    client = create_client()
    chat_only_agent = client.create_agent(
        name="conversation_agent",
        agent_type=AgentType.chat_only_agent,
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=["send_message"],
        memory=conversation_memory,
        include_base_tools=False,
        include_memory_tools=False,
    )
    assert chat_only_agent is not None
    assert chat_only_agent.memory.list_block_labels() == ["chat_agent_persona", "chat_agent_human"]

    for message in ["hello", "my name is not chad, my name is swoodily"]:
        client.send_message(agent_id=chat_only_agent.id, message=message, role="user")
        chat_only_agent = client.get_agent(agent_id=chat_only_agent.id)

    chat_only_agent = client.get_agent(agent_id=chat_only_agent.id)
    assert chat_only_agent.memory.get_block("chat_agent_human").value != get_human_text(DEFAULT_HUMAN)