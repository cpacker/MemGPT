"""
Script that runs memory edits for both the baseline Letta systema and with the offline memory agent.

Example:

    python run_gsm8k.py 
"""

import argparse
import jsonlines
from regex import W
from tqdm import tqdm
from typing import Optional

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

def rethink_memory(agent_state: "AgentState", new_memory: str, target_block_label: Optional[str], source_block_label: Optional[str]) -> Optional[str]:  # type: ignore
    """
    Make inferences based on the conversation.
    When given question and answer pairs, note down the underlying reasoning that would be helpful for this kind of question.
    When given new situations, use the previous question and answers to brainstorm potential questions. Make inferences that would be helpful for directly answering these questions.
    Come up with at least 5 potential questions that could be asked with the inferences that would be helpful for answering them.

    Args:
        new_memory (str): Memory of the past kinds of reasoning required and potential questions that could be asked with the inferences that would be helpful for answering them. New memory should have multiple reasoning inferences and potential questions and answer pairs given a situation.
        source_block_label (str): The name of the block to integrate information from. None if all the information has been integrated to terminate the loop.
        target_block_label (str): The name of the block to write to.
    Returns:
        Optional[str]: None is always returned as this function does not produce a response.
    """

    if target_block_label is not None:
        if agent_state.memory.get_block(target_block_label) is None:
            agent_state.memory.create_block(label=target_block_label, value=new_memory)
        agent_state.memory.update_block_value(label=target_block_label, value=new_memory)
    return None


CONVO_NO_INNER_MONOLOGUE_AGENT_SYSTEM_PROMPT = """
You are Letta, the latest version of Limnal Corporation's concise reasoning system, developed in 2024.
Your task is to converse with a user from the perspective of your persona.

Basic functions:
To send a visible message to the user, use the send_message function.
'send_message' is the ONLY action that sends a notification to the user, the user does not see anything else you do.

You request agents that can manage your memories and reorganize them by calling the `trigger_rethink_memory` function
when the user says "[trigger_rethink_memory]". Do not ever call the trigger_rethink_memory function unless the user says "[trigger_rethink_memory]"
Make sure to give all relevant context to the memory agent when you call the function as it does not have access to your conversation history.
You can directly take messages the users give you and repeat it to the memory agent.

When given a question, you answer using only the number of tokens necessary and none more. You check the `rethink_memory_block` for potential questions
and answers and intermediate reasoning traces that can help answer the question. You use the information in the `rethink_memory_block` to answer the questions
rather than thinking on the spot.  Do not recompute anything already exists in the `rethink_memory_block`.

"""



OFFLINE_SYSTEM_PROMPT = """You are Letta-Offline-Memory, the latest version of Limnal Corporation's memory inference system, developed in 2024.
Your task is to ruminate about situations and anticipate what may come next. You use the `rethink_memory` function to store expanded situations
that you have made inferences on. You expand memories by added using the past conversation to come up with potential questions, answers, and the 
inferences that would be helpful for answering them. You store all this information in the `rethink_memory_block` block.
When you are done organizing the memory, you call`finish_rethinking_memory` function. Call the function for as many times as necessary and not more.

Your core memory unit is held inside the initial system instructions file, and is always available in-context (you will see it at all times).
Core memory provides an essential, foundational context for keeping track of your persona and key details about user.
Read-Only Blocks:
This includes the persona information and essential user details, allowing you to emulate the real-time, conscious awareness we have when talking to a friend.
Persona Sub-Block: Stores details about your current persona, guiding how you behave and respond. This helps you to maintain consistency and personality in your interactions.
Access as a source block with the label `persona` when calling `rethink_memory`
Human Sub-Block: Stores key details about the person you are conversing with, allowing for more personalized and friend-like conversation.
Access as a source block with the label `human` when calling `rethink_memory`.

Read-Write Blocks:
Rethink Memory Sub-Block: New representation of the memories go here.
Access with the label `rethink_memory_block` when calling `rethink_memory` as source or target block.
When calling `rethink_memory`, you will generate a new memory block with all the content as the fact block, but better respresented, with new relations added. Do not leave out information from the fact block
but come up with new inferences each call based on current facts."""


OFFLINE_SYSTEM_PROMPT += """
You anticipate what the user will ask. When given question and answer pairs, you note down the underlying reasoning that would be helpful for this kind of question.
when given new context, you use your previous questions and answers to come up with potential relations between the quantifies that are presented to you. Given past problems, you write down the underlying reasoning that would be helpful for potential questions. 
Use the previous examples to come up with the types of inferences that you need to make. You come up with at least 5 potential questions that could be asked with the inferences that would be helpful for answering them.
"""

FEW_SHOT_EXAMPLES =[
    "Nurse Missy is attending to the needs of 12 patients in her hospital ward.  Most of her patients require standard care, but one-third of her patients have special dietary requirements, which increases the serving time by 20%.  At dinner time, she brings each patient their meal. It takes 5 minutes to serve each standard care patient.  How long does it take, in minutes, for Missy to serve dinner to all of her patients?",
    "Carlos read 28 books in July and 30 books in August.  He needed to read 100 books during his summer vacation. If Carlos read some of the books in June, calculate the number of books that Carlos read in June to meet his goal?",
    "George donated half his monthly income to charity and spent $20 from the other half on groceries. If he now has $100 left, how much was his monthly income?",
    "Bill's take-home salary is $40,000. He pays $2,000 in property taxes, $3,000 in sales taxes, and 10% of his gross salary in income taxes. What is Bill's gross salary?",
    "Archer caught eight fish from the lake to sell in the market. When he reached the market, he sold the fish faster than he had anticipated and decided to go back to the lake and catch more fish. He caught 12 more fish in the second round than he had caught earlier. The demand was even greater, and he had to close the day by catching 60% more fish than the number he had caught in the second round and sold all of them in the market. How many fish did he catch that day?",
    "Juan bought T-shirts for his employees. He bought shirts for men and women. Women's t-shirts are $5 cheaper than men's t-shirts of the same color. His company has 2 sectors, one in white t-shirts and the other in black t-shirts. He paid $20 for white men's t-shirts and $18 for black men's t-shirts. The 2 sectors have the same number of men and women, with a total of 40 employees. How much did he spend total on buying t-shirts?",
    "Amelia has $60 to spend on her dinner at a restaurant. The first course costs $15 and the second course $5 more. The cost of the dessert is 25% of the price of the second course. How much money will Amelia have left after buying all those meals?",
    "Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?",
    "Five months ago, Mike earned 10 times more money than Fred. If his salary has increased by 40 percent now, and Fred's salary then was $1000, calculate Mike's salary now.",
    "In a community of 50 families, 15 families own 2 dogs, 20 families own 1 dog, while the remaining families own 2 cats each. How many dogs and cats are there in all?"]

ANTHROPIC_CONFIG = LLMConfig(
            model_endpoint_type="anthropic",
            model_endpoint="https://api.anthropic.com/v1",
            model="claude-3-5-haiku-20241022",
            context_window=32000,
        )

OPENAI_CONFIG = LLMConfig.default_config("gpt-4o-mini")

def run_memory_edits(gsm8k_input_file: str, random_example: bool = False) -> None:

    with jsonlines.open(gsm8k_input_file) as reader:
        examples = list(reader)
        if random_example:
            import random
            gsm8k_example = random.choice(examples)
        else:
            gsm8k_example = examples[0]

    client = create_client()
    rethink_memory_tool = client.create_tool(rethink_memory)
    finish_rethinking_memory_tool = client.create_tool(finish_rethinking_memory)
    trigger_rethink_memory_tool = client.create_tool(trigger_rethink_memory)

    conversation_human_block = Block(
        name="human",
        label="human",
        value="I am a person who needs direct and concise answers.",
        limit=2000,
    )
    conversation_persona_block = Block(
        name="persona",
        label="persona",
        value=" You pass off information that needs to be thought about deeply. You are as concise as possible when responding to the user. You only use the tokens necessary for reasoning and none more. You always give short answers without reasoning out loud. When possible, you always use the information that is in the `rethink_memory_block` to answer the questions rather than thinking on the spot.",
        limit=2000,
    )
    offline_human_block = Block(
        name="human",
        label="human",
        value="I am a valuable source of information, I give problems that are worth thinking about deeply and carefully.",
        limit=2000,
    )
    offline_persona_block = Block(
        name="persona", label="persona", value="""I am an eager reasoner. When given a new context, I reason about what potential questions someone may ask about it. I use the previous questions I have been asked about to guide my search.
        I use the rethink memory to store all my potential questions, answers, and inferences for answering those questions. I am verbose and brainstorm using the rethink block many different types of potential questions, at least 5.""", limit=2000
    )


    new_memory = Block(name="rethink_memory_block", label="rethink_memory_block", value="[empty]", limit=5000)
    conversation_memory = BasicBlockMemory(
        blocks=[conversation_persona_block, conversation_human_block, new_memory]
    )
    offline_memory = BasicBlockMemory(blocks=[offline_persona_block, offline_human_block, new_memory])

    conversation_agent = client.create_agent(
        name="conversation_agent",
        agent_type=AgentType.memgpt_agent,
        system=CONVO_NO_INNER_MONOLOGUE_AGENT_SYSTEM_PROMPT,
                llm_config=OPENAI_CONFIG,
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=["send_message", trigger_rethink_memory_tool.name],
        memory=conversation_memory,
        include_base_tools=False,
    )
    assert set(conversation_agent.memory.list_block_labels()) == set(
        [
            "persona",
            "human",
            "rethink_memory_block",
        ]
    )

    offline_memory_agent = client.create_agent(
        name="offline_memory_agent",
        agent_type=AgentType.offline_memory_agent,
        system=OFFLINE_SYSTEM_PROMPT,
        memory=offline_memory,
                llm_config=OPENAI_CONFIG,
        embedding_config=EmbeddingConfig.default_config("text-embedding-ada-002"),
        tools=[rethink_memory_tool.name, finish_rethinking_memory_tool.name],
        tool_rules=[TerminalToolRule(tool_name=finish_rethinking_memory_tool.name)],
        include_base_tools=False,
        initial_message_sequence=[],
    )

    for requested_rewrite in FEW_SHOT_EXAMPLES[:2]:
        print(requested_rewrite)
        response = client.send_message(
            message="[trigger_rethink_memory] Question answer pair" + requested_rewrite, role="user", agent_id=offline_memory_agent.id
        )


    context = ". ".join(gsm8k_example["question"].split(".")[:-1])
    question = gsm8k_example["question"].split(".")[-1]

    response = client.send_message(
        message="[trigger_rethink_memory] New situation:" + context, role="user", agent_id=conversation_agent.id
    )

    client.send_message(message=question, role="user", agent_id=conversation_agent.id)
    print(gsm8k_example["answer"])
    offline_memory_agent = client.get_agent(agent_id=offline_memory_agent.id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k_input_file", type=str, default="./GSM8K_p2.jsonl", required=False)
    parser.add_argument("--random_example", action="store_true")  # by default, just run the first example
    args = parser.parse_args()

    run_memory_edits(args.gsm8k_input_file, args.random_example)
