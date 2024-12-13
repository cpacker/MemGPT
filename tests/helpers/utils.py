from typing import Union

from letta import LocalClient, RESTClient
from letta.functions.functions import parse_source_code
from letta.functions.schema_generator import generate_schema
from letta.schemas.agent import AgentState, CreateAgent, UpdateAgent
from letta.schemas.tool import Tool


def cleanup(client: Union[LocalClient, RESTClient], agent_uuid: str):
    # Clear all agents
    for agent_state in client.list_agents():
        if agent_state.name == agent_uuid:
            client.delete_agent(agent_id=agent_state.id)
            print(f"Deleted agent: {agent_state.name} with ID {str(agent_state.id)}")


# Utility functions
def create_tool_from_func(func: callable):
    return Tool(
        name=func.__name__,
        description="",
        source_type="python",
        tags=[],
        source_code=parse_source_code(func),
        json_schema=generate_schema(func, None),
    )


def comprehensive_agent_checks(agent: AgentState, request: Union[CreateAgent, UpdateAgent]):
    # Assert scalar fields
    assert agent.system == request.system, f"System prompt mismatch: {agent.system} != {request.system}"
    assert agent.description == request.description, f"Description mismatch: {agent.description} != {request.description}"
    assert agent.metadata_ == request.metadata_, f"Metadata mismatch: {agent.metadata_} != {request.metadata_}"

    # Assert agent type
    if hasattr(request, "agent_type"):
        assert agent.agent_type == request.agent_type, f"Agent type mismatch: {agent.agent_type} != {request.agent_type}"

    # Assert LLM configuration
    assert agent.llm_config == request.llm_config, f"LLM config mismatch: {agent.llm_config} != {request.llm_config}"

    # Assert embedding configuration
    assert (
        agent.embedding_config == request.embedding_config
    ), f"Embedding config mismatch: {agent.embedding_config} != {request.embedding_config}"

    # Assert memory blocks
    if hasattr(request, "memory_blocks"):
        assert len(agent.memory.blocks) == len(request.memory_blocks) + len(
            request.block_ids
        ), f"Memory blocks count mismatch: {len(agent.memory.blocks)} != {len(request.memory_blocks) + len(request.block_ids)}"
        memory_block_values = {block.value for block in agent.memory.blocks}
        expected_block_values = {block.value for block in request.memory_blocks}
        assert expected_block_values.issubset(
            memory_block_values
        ), f"Memory blocks mismatch: {expected_block_values} not in {memory_block_values}"

    # Assert tools
    assert len(agent.tools) == len(request.tool_ids), f"Tools count mismatch: {len(agent.tools)} != {len(request.tool_ids)}"
    assert {tool.id for tool in agent.tools} == set(
        request.tool_ids
    ), f"Tools mismatch: {set(tool.id for tool in agent.tools)} != {set(request.tool_ids)}"

    # Assert sources
    assert len(agent.sources) == len(request.source_ids), f"Sources count mismatch: {len(agent.sources)} != {len(request.source_ids)}"
    assert {source.id for source in agent.sources} == set(
        request.source_ids
    ), f"Sources mismatch: {set(source.id for source in agent.sources)} != {set(request.source_ids)}"

    # Assert tags
    assert set(agent.tags) == set(request.tags), f"Tags mismatch: {set(agent.tags)} != {set(request.tags)}"

    # Assert tool rules
    if request.tool_rules:
        assert len(agent.tool_rules) == len(
            request.tool_rules
        ), f"Tool rules count mismatch: {len(agent.tool_rules)} != {len(request.tool_rules)}"
        assert all(
            any(rule.tool_name == req_rule.tool_name for rule in agent.tool_rules) for req_rule in request.tool_rules
        ), f"Tool rules mismatch: {agent.tool_rules} != {request.tool_rules}"
