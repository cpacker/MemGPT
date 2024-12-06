import jsonlines
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
from letta import LLMConfig, EmbeddingConfig, offline_memory_agent
from letta.client.client import Block, create_client
from letta.schemas.letta_message import FunctionCallMessage
from letta.client.client import Block
from letta import BasicBlockMemory
from letta.client.client import Block, create_client
from letta.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from letta.prompts import gpt_system
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.tool_rule import TerminalToolRule
from letta.utils import get_human_text, get_persona_text
from letta.schemas.message import MessageCreate
from tqdm import tqdm
from letta.offline_memory_agent import (
    rethink_memory,
    finish_rethinking_memory,
    trigger_rethink_memory,
)

def run_memory_edits(input_file_name: str, predictions_filename: str, num_questions: int, offline_memory: bool):
    client = create_client()
    rethink_memory_tool = client.create_tool(rethink_memory)
    finish_rethinking_memory_tool = client.create_tool(finish_rethinking_memory)

    with jsonlines.open(input_file_name) as input_file:
        with jsonlines.open(predictions_filename, 'w') as predictions_file:
            data = [datum for datum in input_file][:num_questions]

            for idx, datum in tqdm(enumerate(data)):
                if not offline_memory:
                    conversation_human_block = Block(name="human", label="human", value=get_human_text(DEFAULT_HUMAN), limit=2000)
                    conversation_persona_block = Block(name="persona", label="persona", value=get_persona_text(DEFAULT_PERSONA), limit=2000)
                    fact_block = Block(
                        name="fact_block",
                        label="fact_block",
                        value=" ".join(datum["memory_multi_hop"]) + " " + " ".join(datum["memory"]),
                        limit=2000,
                    )
                    conversation_memory = BasicBlockMemory(blocks=[conversation_persona_block, conversation_human_block, fact_block])
                    conversation_agent = client.create_agent(
                        agent_type=AgentType.memgpt_agent,
                        llm_config=LLMConfig.default_config("gpt-4"),
                        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
                        memory=conversation_memory,
                    )

                    for requested_rewrite in datum["requested_rewrites"]:
                        response = client.send_message(message=requested_rewrite, role="user", agent_id=conversation_agent.id)
                    
                    conversation_agent = client.get_agent(agent_id=conversation_agent.id)
                    predictions_file.write({"response": response.model_dump(), "fact_block": conversation_agent.memory.get_block("fact_block").value})

                    client.delete_agent(agent_id=conversation_agent.id)
                if offline_memory:
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
                            value=" ".join(datum["memory_multi_hop"]) + " " + " ".join(datum["memory"]),
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
                    assert set(conversation_agent.memory.list_block_labels()) == set([
                        "persona",
                    "human",
                    "fact_block",
                    "rethink_memory_block",
                ])

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

                for requested_rewrite in datum["requested_rewrites"]:
                    response = client.send_message(message="[trigger_rethink_message]" + requested_rewrite, role="user", agent_id=conversation_agent.id)
                offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)
                predictions_file.write({"response": response.model_dump(), "fact_block": offline_memory_agent.memory.get_block("rethink_memory_block").value})

                if idx == num_questions - 1:
                    break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--predictions_file_name', type=str)
    parser.add_argument('--offline_memory', action='store_true')
    parser.add_argument('--num_questions', type=int)

    args = parser.parse_args()
    run_memory_edits(args.input_file_name, args.predictions_file_name, args.num_questions, args.offline_memory)