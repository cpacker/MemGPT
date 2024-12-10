"""
Script that runs memory edits for both the baseline Letta systema and with the offline memory agent.

Example:

    python run_fact_editing.py --input_file_name MQuAKE/datasets/letta-MQuAKE-CF-3k-v2-100.json \
        --predictions_file_name MQuaAKE/datasets/debug-letta-MQuAKE-CF-3k-v2-100-predictions.json \
        --num_questions 1000


"""

import argparse

import jsonlines
from tqdm import tqdm

from letta import BasicBlockMemory, EmbeddingConfig, LLMConfig
from letta.client.client import Block, create_client
from letta.offline_memory_agent import (
    finish_rethinking_memory,
    rethink_memory,
    trigger_rethink_memory,
)
from letta.prompts import gpt_system
from letta.schemas.agent import AgentType
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig
from letta.schemas.tool_rule import TerminalToolRule
from letta.utils import get_persona_text

OFFLINE_SYSTEM_PROMPT = """You are Letta-Offline-Memory, the latest version of Limnal Corporation's digital companion, developed in 2024.
Your task is to re-organize and consolidate memories by calling `rethink_memory` at every single step, when you are done reorganizing the memory, you use the
`finish_rethinking_memory` function. Call the function for as many times as necessary and not more.
Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides an essential, foundational context for keeping track of your persona and key details about user.
Read-Only Blocks:
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.
Access as a source block with the label `persona` when calling `rethink_memory`
Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
Access as a source block with the label `human` when calling `rethink_memory`.
Read-Write Blocks:
Rethink Memory Sub-Block: New representation of the memories go here. Access with the label `rethink_memory_block` when calling `rethink_memory` as source or target block.
Do not remove information unless it has been replaced with new information. Use language as close to what is in the block as possible.
At every step, you reorganize the memories by calling the `rethink_memory` function. You use this to take current information in the `rethink_memory` block and select a single memory block to integrate information from, producing a new memory for the rethink_memory_block.  The new memory is the result
of new insights, and new inferences and hypotheses based on the past memories. Make sure to consider how the new information affects each memory.
Prioritize the new information overy existing memories. If the new information implies that the old memory may need to change, then output the most
likely fact given the update information. Given new information and your current memory, you draw all logical conclusions and potential hypotheses possible with the `rethink_memory` function.
If you are uncertain, use your internal monologue to consider what the possible conclusions are, and then state the most likely new facts that would replace the old facts in the new memory block.
"""


def run_memory_edits(input_file_name: str, predictions_filename: str, num_questions: int, offline_memory: bool):
    client = create_client()
    rethink_memory_tool = client.create_tool(rethink_memory)
    finish_rethinking_memory_tool = client.create_tool(finish_rethinking_memory)
    trigger_rethink_memory_tool = client.create_tool(trigger_rethink_memory)

    with jsonlines.open(input_file_name) as input_file:
        with jsonlines.open(predictions_filename, "w") as predictions_file:
            data = [datum for datum in input_file][:num_questions]

            for idx, datum in tqdm(enumerate(data)):
                if not offline_memory:
                    conversation_human_block = Block(
                        name="human",
                        label="human",
                        value="I am a trustworthy new source. I give verified updates of new facts. I am to be trusted as a source of truth.",
                        limit=2000,
                    )
                    conversation_persona_block = Block(
                        name="persona",
                        label="persona",
                        value=" You are a counterfactual reasoner. You listen are given trustworthy factual updates. You always trust the new information no matter what, even if it is against you current knowledge of the world.",
                        limit=2000,
                    )
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

                    predictions_file.write(
                        {
                            "case_id": datum["case_id"],
                            "messages": [message.to_openai_dict() for message in client.get_messages(conversation_agent.id)],
                            "memory_blocks": {block.label: block.value for block in conversation_agent.memory.get_blocks()},
                        }
                    )

                    client.delete_agent(agent_id=conversation_agent.id)
                if offline_memory:
                    conversation_human_block = Block(
                        name="human",
                        label="human",
                        value="I am a trustworthy new source. I give verified updates of new facts. I am to be trusted as a source of truth.",
                        limit=2000,
                    )
                    conversation_persona_block = Block(
                        name="persona",
                        label="persona",
                        value=" You are a counterfactual reasoner. You listen are given trustworthy factual updates. You always trust the new information no matter what, even if it is against you current knowledge of the world.",
                        limit=2000,
                    )
                    offline_human_block = Block(
                        name="human",
                        label="human",
                        value="I am a trustworthy new source. I give verified updates of new facts. I am to be trusted as a source of truth.",
                        limit=2000,
                    )
                    offline_persona_block = Block(
                        name="persona", label="persona", value=get_persona_text("offline_memory_persona"), limit=2000
                    )
                    fact_block = Block(
                        name="fact_block",
                        label="fact_block",
                        value=" ".join(datum["memory_multi_hop"]) + " " + " ".join(datum["memory"]),
                        limit=2000,
                    )

                    new_memory = Block(name="rethink_memory_block", label="rethink_memory_block", value="[empty]", limit=2000)
                    conversation_memory = BasicBlockMemory(
                        blocks=[conversation_persona_block, conversation_human_block, fact_block, new_memory]
                    )
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
                    assert set(conversation_agent.memory.list_block_labels()) == set(
                        [
                            "persona",
                            "human",
                            "fact_block",
                            "rethink_memory_block",
                        ]
                    )

                    offline_memory_agent = client.create_agent(
                        name="offline_memory_agent",
                        agent_type=AgentType.offline_memory_agent,
                        system=OFFLINE_SYSTEM_PROMPT,
                        memory=offline_memory,
                        llm_config=LLMConfig.default_config("gpt-4"),
                        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
                        tools=[rethink_memory_tool.name, finish_rethinking_memory_tool.name],
                        tool_rules=[TerminalToolRule(tool_name=finish_rethinking_memory_tool.name)],
                        include_base_tools=False,
                    )

                    for requested_rewrite in datum["requested_rewrites"]:
                        response = client.send_message(
                            message="[trigger_rethink_message]" + requested_rewrite, role="system", agent_id=conversation_agent.id
                        )
                    offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)

                    predictions_file.write(
                        {
                            "case_id": datum["case_id"],
                            "conversation_agent_messages": [
                                message.to_openai_dict() for message in client.get_messages(conversation_agent.id)
                            ],
                            "conversation_agent_memory_blocks": {
                                block.label: block.value for block in conversation_agent.memory.get_blocks()
                            },
                            "offline_memory_agent_messages": [
                                message.to_openai_dict() for message in client.get_messages(offline_memory_agent.id)
                            ],
                            "offline_memory_agent_memory": {block.label: block.value for block in offline_memory_agent.memory.get_blocks()},
                        }
                    )
                    client.delete_agent(agent_id=conversation_agent.id)
                    client.delete_agent(agent_id=offline_memory_agent.id)

                if idx == num_questions - 1:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", type=str)
    parser.add_argument("--predictions_file_name", type=str)
    parser.add_argument("--offline_memory", action="store_true")
    parser.add_argument("--num_questions", type=int)

    args = parser.parse_args()
    run_memory_edits(args.input_file_name, args.predictions_file_name, args.num_questions, args.offline_memory)
