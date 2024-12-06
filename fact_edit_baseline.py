import jsonlines
import json
import argparse
from tqdm import tqdm
from copy import deepcopy
from letta import LLMConfig, EmbeddingConfig
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

def run_memory_edits(input_file_name: str, predictions_filename: str, num_questions: int):
    client = create_client()
    with jsonlines.open(input_file_name) as input_file:
        with jsonlines.open(predictions_filename, 'w') as predictions_file:
            data = [datum for datum in input_file][:num_questions]

            for idx, datum in tqdm(enumerate(data)):
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
                if idx == num_questions - 1:
                    break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_name', type=str)
    parser.add_argument('--predictions_file_name', type=str)
    parser.add_argument('--num_questions', type=int)

    args = parser.parse_args()
    run_memory_edits(args.input_file_name, args.predictions_file_name, args.num_questions)