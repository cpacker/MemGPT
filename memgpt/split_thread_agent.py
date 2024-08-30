import datetime
import inspect
import traceback
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

from memgpt.agent import Agent
from memgpt.agent_store.storage import StorageConnector
from memgpt.constants import (
    CLI_WARNING_PREFIX,
    FIRST_MESSAGE_ATTEMPTS,
    IN_CONTEXT_MEMORY_KEYWORD,
    LLM_MAX_TOKENS,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
    MESSAGE_SUMMARY_WARNING_FRAC,
)
from memgpt.interface import AgentInterface
from memgpt.llm_api.llm_api_tools import create, is_context_overflow_error
from memgpt.memory import ArchivalMemory, RecallMemory, summarize_messages
from memgpt.metadata import MetadataStore
from memgpt.persistence_manager import LocalStateManager
from memgpt.schemas.agent import AgentState
from memgpt.schemas.block import Block
from memgpt.schemas.embedding_config import EmbeddingConfig
from memgpt.schemas.enums import OptionState
from memgpt.schemas.memory import Memory
from memgpt.schemas.message import Message
from memgpt.schemas.openai.chat_completion_response import ChatCompletionResponse
from memgpt.schemas.openai.chat_completion_response import (
    Message as ChatCompletionMessage,
)
from memgpt.schemas.passage import Passage
from memgpt.schemas.tool import Tool
from memgpt.system import (
    get_initial_boot_messages,
    get_login_event,
    package_function_response,
    package_summarize_message,
)
from memgpt.utils import (
    count_tokens,
    get_local_time,
    get_tool_call_id,
    get_utc_time,
    is_utc_datetime,
    json_dumps,
    json_loads,
    parse_json,
    printd,
    united_diff,
    validate_function_response,
    verify_first_message_correctness,
)


class AbstractAgent:
    """
    Abstract class for conversational agents.
    """

    def step(
        self,
        user_message: Union[Message, str],  # NOTE: should be json.dump(dict)
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,  # if True, return dicts, if False, return Message objects
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
        stream: bool = False,  # TODO move to config?
        timestamp: Optional[datetime.datetime] = None,
        inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
        ms: Optional[MetadataStore] = None,
    ) -> Tuple[List[Union[dict, Message]], bool, bool, bool]:
        """
        Top-level event message handler for the agent.
        """
        raise NotImplementedError


class SplitThreadAgent(AbstractAgent):
    def __init__(
        self,
        interface: AgentInterface,
        # agents can be created from providing agent_state
        agent_state: AgentState,
        tools: List[Tool],
        # memory: Memory,
        # extras
        messages_total: Optional[int] = None,  # TODO remove?
        first_message_verify_mono: bool = True,  # TODO move to config?
    ):
        self.conversational_agent = Agent(
            interface=interface,
            agent_state=agent_state,
            tools=tools,
            messages_total=messages_total,
            first_message_verify_mono=first_message_verify_mono,
        )

        print("THIS AGENT STATE HAS TOOLS:", agent_state.tools)

        # self.conversational_agent_state = AgentState()
        # self.conversational_agent = Agent()

        # self.memory_agent_state = AgentState()
        # self.memory_agent = Agent()
        #

    @property
    def messages(self) -> List[dict]:
        return self.conversational_agent.messages

    @messages.setter
    def messages(self, value: List[dict]):
        raise ValueError("Cannot set messages directly on SplitThreadAgent")

    def update_state(self) -> AgentState:
        message_ids = [msg.id for msg in self.conversational_agent._messages]
        assert isinstance(self.conversational_agent.memory, Memory), f"Memory is not a Memory object: {type(self.memory)}"

        # override any fields that may have been updated
        self.agent_state.message_ids = message_ids
        self.agent_state.memory = self.memory
        self.agent_state.system = self.system

        return self.agent_state

    def step(
        self,
        user_message: Union[Message, str],  # NOTE: should be json.dump(dict)
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        return_dicts: bool = True,  # if True, return dicts, if False, return Message objects
        recreate_message_timestamp: bool = True,  # if True, when input is a Message type, recreated the 'created_at' field
        stream: bool = False,  # TODO move to config?
        timestamp: Optional[datetime.datetime] = None,
        inner_thoughts_in_kwargs: OptionState = OptionState.DEFAULT,
        ms: Optional[MetadataStore] = None,
    ) -> Tuple[List[Union[dict, Message]], bool, bool, bool]:
        return [], False, False, False

        """Top-level event message handler for the MemGPT agent"""

        def strip_name_field_from_user_message(user_message_text: str) -> Tuple[str, Optional[str]]:
            """If 'name' exists in the JSON string, remove it and return the cleaned text + name value"""
            try:
                user_message_json = dict(json_loads(user_message_text))
                # Special handling for AutoGen messages with 'name' field
                # Treat 'name' as a special field
                # If it exists in the input message, elevate it to the 'message' level
                name = user_message_json.pop("name", None)
                clean_message = json_dumps(user_message_json)

            except Exception as e:
                print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed with: {e}")

            return clean_message, name

        def validate_json(user_message_text: str, raise_on_error: bool) -> str:
            try:
                user_message_json = dict(json_loads(user_message_text))
                user_message_json_val = json_dumps(user_message_json)
                return user_message_json_val
            except Exception as e:
                print(f"{CLI_WARNING_PREFIX}couldn't parse user input message as JSON: {e}")
                if raise_on_error:
                    raise e

        try:
            # Step 0: update core memory
            # only pulling latest block data if shared memory is being used
            # TODO: ensure we're passing in metadata store from all surfaces
            if ms is not None:
                should_update = False
                for block in self.agent_state.memory.to_dict().values():
                    if not block.get("template", False):
                        should_update = True
                if should_update:
                    # TODO: the force=True can be optimized away
                    # once we ensure we're correctly comparing whether in-memory core
                    # data is different than persisted core data.
                    self.rebuild_memory(force=True, ms=ms)
            # Step 1: add user message
            if user_message is not None:
                if isinstance(user_message, Message):
                    # Validate JSON via save/load
                    user_message_text = validate_json(user_message.text, False)
                    cleaned_user_message_text, name = strip_name_field_from_user_message(user_message_text)

                    if name is not None:
                        # Update Message object
                        user_message.text = cleaned_user_message_text
                        user_message.name = name

                    # Recreate timestamp
                    if recreate_message_timestamp:
                        user_message.created_at = get_utc_time()

                elif isinstance(user_message, str):
                    # Validate JSON via save/load
                    user_message = validate_json(user_message, False)
                    cleaned_user_message_text, name = strip_name_field_from_user_message(user_message)

                    # If user_message['name'] is not None, it will be handled properly by dict_to_message
                    # So no need to run strip_name_field_from_user_message

                    # Create the associated Message object (in the database)
                    user_message = Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={"role": "user", "content": cleaned_user_message_text, "name": name},
                        created_at=timestamp,
                    )

                else:
                    raise ValueError(f"Bad type for user_message: {type(user_message)}")

                self.interface.user_message(user_message.text, msg_obj=user_message)

                input_message_sequence = self._messages + [user_message]
            # Alternatively, the requestor can send an empty user message
            else:
                input_message_sequence = self._messages

            if len(input_message_sequence) > 1 and input_message_sequence[-1].role != "user":
                printd(f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue")

            # Step 2: send the conversation and available functions to GPT
            if not skip_verify and (first_message or self.messages_total == self.messages_total_init):
                printd(f"This is the first message. Running extra verifier on AI response.")
                counter = 0
                while True:
                    response = self._get_ai_reply(
                        message_sequence=input_message_sequence,
                        first_message=True,  # passed through to the prompt formatter
                        stream=stream,
                        inner_thoughts_in_kwargs=inner_thoughts_in_kwargs,
                    )
                    if verify_first_message_correctness(response, require_monologue=self.first_message_verify_mono):
                        break

                    counter += 1
                    if counter > first_message_retry_limit:
                        raise Exception(f"Hit first message retry limit ({first_message_retry_limit})")

            else:
                response = self._get_ai_reply(
                    message_sequence=input_message_sequence,
                    stream=stream,
                    inner_thoughts_in_kwargs=inner_thoughts_in_kwargs,
                )

            # Step 3: check if LLM wanted to call a function
            # (if yes) Step 4: call the function
            # (if yes) Step 5: send the info on the function call and function response to LLM
            response_message = response.choices[0].message
            response_message.model_copy()  # TODO why are we copying here?
            all_response_messages, heartbeat_request, function_failed = self._handle_ai_response(
                response_message,
                # TODO this is kind of hacky, find a better way to handle this
                # the only time we set up message creation ahead of time is when streaming is on
                response_message_id=response.id if stream else None,
            )

            # Add the extra metadata to the assistant response
            # (e.g. enough metadata to enable recreating the API call)
            # assert "api_response" not in all_response_messages[0]
            # all_response_messages[0]["api_response"] = response_message_copy
            # assert "api_args" not in all_response_messages[0]
            # all_response_messages[0]["api_args"] = {
            #     "model": self.model,
            #     "messages": input_message_sequence,
            #     "functions": self.functions,
            # }

            # Step 6: extend the message history
            if user_message is not None:
                if isinstance(user_message, Message):
                    all_new_messages = [user_message] + all_response_messages
                else:
                    raise ValueError(type(user_message))
            else:
                all_new_messages = all_response_messages

            # Check the memory pressure and potentially issue a memory pressure warning
            current_total_tokens = response.usage.total_tokens
            active_memory_warning = False
            # We can't do summarize logic properly if context_window is undefined
            if self.agent_state.llm_config.context_window is None:
                # Fallback if for some reason context_window is missing, just set to the default
                print(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
                print(f"{self.agent_state}")
                self.agent_state.llm_config.context_window = (
                    LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
                )
            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window):
                printd(
                    f"{CLI_WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )
                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this
            else:
                printd(
                    f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * int(self.agent_state.llm_config.context_window)}"
                )

            self._append_to_messages(all_new_messages)
            messages_to_return = [msg.to_openai_dict() for msg in all_new_messages] if return_dicts else all_new_messages

            # update state after each step
            self.update_state()

            return messages_to_return, heartbeat_request, function_failed, active_memory_warning, response.usage

        except Exception as e:
            printd(f"step() failed\nuser_message = {user_message}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                # A separate API call to run a summarizer
                self.summarize_messages_inplace()

                # Try step again
                return self.step(
                    user_message,
                    first_message=first_message,
                    first_message_retry_limit=first_message_retry_limit,
                    skip_verify=skip_verify,
                    return_dicts=return_dicts,
                    recreate_message_timestamp=recreate_message_timestamp,
                    stream=stream,
                    timestamp=timestamp,
                    inner_thoughts_in_kwargs=inner_thoughts_in_kwargs,
                    ms=ms,
                )

            else:
                printd(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e
