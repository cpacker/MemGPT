from typing import Callable, Dict, List

from letta.constants import MESSAGE_SUMMARY_REQUEST_ACK, MESSAGE_SUMMARY_WARNING_FRAC
from letta.llm_api.llm_api_tools import create
from letta.prompts.gpt_summarize import SYSTEM as SUMMARY_PROMPT_SYSTEM
from letta.schemas.agent import AgentState
from letta.schemas.enums import MessageRole
from letta.schemas.memory import Memory
from letta.schemas.message import Message
from letta.utils import count_tokens, printd


def get_memory_functions(cls: Memory) -> Dict[str, Callable]:
    """Get memory functions for a memory class"""
    functions = {}

    # collect base memory functions (should not be included)
    base_functions = []
    for func_name in dir(Memory):
        funct = getattr(Memory, func_name)
        if callable(funct):
            base_functions.append(func_name)

    for func_name in dir(cls):
        if func_name.startswith("_") or func_name in ["load", "to_dict"]:  # skip base functions
            continue
        if func_name in base_functions:  # dont use BaseMemory functions
            continue
        func = getattr(cls, func_name)
        if not callable(func):  # not a function
            continue
        functions[func_name] = func
    return functions


def _format_summary_history(message_history: List[Message]):
    # TODO use existing prompt formatters for this (eg ChatML)
    return "\n".join([f"{m.role}: {m.text}" for m in message_history])


def summarize_messages(
    agent_state: AgentState,
    message_sequence_to_summarize: List[Message],
):
    """Summarize a message sequence using GPT"""
    # we need the context_window
    context_window = agent_state.llm_config.context_window

    summary_prompt = SUMMARY_PROMPT_SYSTEM
    summary_input = _format_summary_history(message_sequence_to_summarize)
    summary_input_tkns = count_tokens(summary_input)
    if summary_input_tkns > MESSAGE_SUMMARY_WARNING_FRAC * context_window:
        trunc_ratio = (MESSAGE_SUMMARY_WARNING_FRAC * context_window / summary_input_tkns) * 0.8  # For good measure...
        cutoff = int(len(message_sequence_to_summarize) * trunc_ratio)
        summary_input = str(
            [summarize_messages(agent_state, message_sequence_to_summarize=message_sequence_to_summarize[:cutoff])]
            + message_sequence_to_summarize[cutoff:]
        )

    dummy_agent_id = agent_state.id
    message_sequence = []
    message_sequence.append(Message(agent_id=dummy_agent_id, role=MessageRole.system, text=summary_prompt))
    message_sequence.append(Message(agent_id=dummy_agent_id, role=MessageRole.assistant, text=MESSAGE_SUMMARY_REQUEST_ACK))
    message_sequence.append(Message(agent_id=dummy_agent_id, role=MessageRole.user, text=summary_input))

    # TODO: We need to eventually have a separate LLM config for the summarizer LLM
    llm_config_no_inner_thoughts = agent_state.llm_config.model_copy(deep=True)
    llm_config_no_inner_thoughts.put_inner_thoughts_in_kwargs = False
    response = create(
        llm_config=llm_config_no_inner_thoughts,
        user_id=agent_state.created_by_id,
        messages=message_sequence,
        stream=False,
    )

    printd(f"summarize_messages gpt reply: {response.choices[0]}")
    reply = response.choices[0].message.content
    return reply
