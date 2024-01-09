import datetime
import uuid
import glob
import os
import json
import traceback

from memgpt.data_types import AgentState
from memgpt.metadata import MetadataStore
from memgpt.interface import AgentInterface
from memgpt.persistence_manager import PersistenceManager, LocalStateManager
from memgpt.config import MemGPTConfig
from memgpt.system import get_login_event, package_function_response, package_summarize_message, get_initial_boot_messages
from memgpt.memory import CoreMemory as InContextMemory, summarize_messages
from memgpt.openai_tools import create, is_context_overflow_error
from memgpt.utils import (
    get_local_time,
    parse_json,
    united_diff,
    printd,
    count_tokens,
    get_schema_diff,
    validate_function_response,
    verify_first_message_correctness,
)
from memgpt.constants import (
    FIRST_MESSAGE_ATTEMPTS,
    MESSAGE_SUMMARY_WARNING_FRAC,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    CORE_MEMORY_HUMAN_CHAR_LIMIT,
    CORE_MEMORY_PERSONA_CHAR_LIMIT,
    LLM_MAX_TOKENS,
    CLI_WARNING_PREFIX,
)
from .errors import LLMError
from .functions.functions import load_all_function_sets


def link_functions(function_schemas):
    """Link function definitions to list of function schemas"""

    # need to dynamically link the functions
    # the saved agent.functions will just have the schemas, but we need to
    # go through the functions library and pull the respective python functions

    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    # agent.functions is a list of schemas (OpenAI kwarg functions style, see: https://platform.openai.com/docs/api-reference/chat/create)
    # [{'name': ..., 'description': ...}, {...}]
    available_functions = load_all_function_sets()
    linked_function_set = {}
    for f_schema in function_schemas:
        # Attempt to find the function in the existing function library
        f_name = f_schema.get("name")
        if f_name is None:
            raise ValueError(f"While loading agent.state.functions encountered a bad function schema object with no name:\n{f_schema}")
        linked_function = available_functions.get(f_name)
        if linked_function is None:
            raise ValueError(
                f"Function '{f_name}' was specified in agent.state.functions, but is not in function library:\n{available_functions.keys()}"
            )
        # Once we find a matching function, make sure the schema is identical
        if json.dumps(f_schema, ensure_ascii=False) != json.dumps(linked_function["json_schema"], ensure_ascii=False):
            # error_message = (
            #     f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different."
            #     + f"\n>>>agent.state.functions\n{json.dumps(f_schema, indent=2, ensure_ascii=False)}"
            #     + f"\n>>>function library\n{json.dumps(linked_function['json_schema'], indent=2, ensure_ascii=False)}"
            # )
            schema_diff = get_schema_diff(f_schema, linked_function["json_schema"])
            error_message = (
                f"Found matching function '{f_name}' from agent.state.functions inside function library, but schemas are different.\n"
                + "".join(schema_diff)
            )

            # NOTE to handle old configs, instead of erroring here let's just warn
            # raise ValueError(error_message)
            printd(error_message)
        linked_function_set[f_name] = linked_function
    return linked_function_set


def initialize_memory(ai_notes, human_notes):
    if ai_notes is None:
        raise ValueError(ai_notes)
    if human_notes is None:
        raise ValueError(human_notes)
    memory = InContextMemory(human_char_limit=CORE_MEMORY_HUMAN_CHAR_LIMIT, persona_char_limit=CORE_MEMORY_PERSONA_CHAR_LIMIT)
    memory.edit_persona(ai_notes)
    memory.edit_human(human_notes)
    return memory


def construct_system_with_memory(system, memory, memory_edit_timestamp, archival_memory=None, recall_memory=None, include_char_count=True):
    full_system_message = "\n".join(
        [
            system,
            "\n",
            f"### Memory [last modified: {memory_edit_timestamp.strip()}]",
            f"{len(recall_memory) if recall_memory else 0} previous messages between you and the user are stored in recall memory (use functions to access them)",
            f"{len(archival_memory) if archival_memory else 0} total memories you created are stored in archival memory (use functions to access them)",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
            f'<persona characters="{len(memory.persona)}/{memory.persona_char_limit}">' if include_char_count else "<persona>",
            memory.persona,
            "</persona>",
            f'<human characters="{len(memory.human)}/{memory.human_char_limit}">' if include_char_count else "<human>",
            memory.human,
            "</human>",
        ]
    )
    return full_system_message


def initialize_message_sequence(
    model,
    system,
    memory,
    archival_memory=None,
    recall_memory=None,
    memory_edit_timestamp=None,
    include_initial_boot_message=True,
):
    if memory_edit_timestamp is None:
        memory_edit_timestamp = get_local_time()

    full_system_message = construct_system_with_memory(
        system, memory, memory_edit_timestamp, archival_memory=archival_memory, recall_memory=recall_memory
    )
    first_user_message = get_login_event()  # event letting MemGPT know the user just logged in

    if include_initial_boot_message:
        if model is not None and "gpt-3.5" in model:
            initial_boot_messages = get_initial_boot_messages("startup_with_send_message_gpt35")
        else:
            initial_boot_messages = get_initial_boot_messages("startup_with_send_message")
        messages = (
            [
                {"role": "system", "content": full_system_message},
            ]
            + initial_boot_messages
            + [
                {"role": "user", "content": first_user_message},
            ]
        )

    else:
        messages = [
            {"role": "system", "content": full_system_message},
            {"role": "user", "content": first_user_message},
        ]

    return messages


class Agent(object):
    def __init__(
        self,
        agent_state: AgentState,
        interface: AgentInterface,
        # extras
        messages_total=None,  # TODO remove?
        first_message_verify_mono=True,  # TODO move to config?
    ):
        # Hold a copy of the state that was used to init the agent
        self.config = agent_state  # TODO: remove
        self.agent_state = agent_state

        # gpt-4, gpt-3.5-turbo, ...
        self.model = agent_state.llm_config.model

        # Store the system instructions (used to rebuild memory)
        if "system" not in agent_state.state:
            raise ValueError(f"'system' not found in provided AgentState")
        self.system = agent_state.state["system"]

        if "functions" not in agent_state.state:
            raise ValueError(f"'functions' not found in provided AgentState")
        # Store the functions schemas (this is passed as an argument to ChatCompletion)
        self.functions = agent_state.state["functions"]  # these are the schema
        # Link the actual python functions corresponding to the schemas
        self.functions_python = {k: v["python_function"] for k, v in link_functions(function_schemas=self.functions).items()}
        assert all([callable(f) for k, f in self.functions_python.items()]), self.functions_python

        # Initialize the memory object
        if "persona" not in agent_state.state:
            raise ValueError(f"'persona' not found in provided AgentState")
        if "human" not in agent_state.state:
            raise ValueError(f"'human' not found in provided AgentState")
        self.memory = initialize_memory(ai_notes=agent_state.state["persona"], human_notes=agent_state.state["human"])
        # Once the memory object is initialize, use it to "bake" the system message
        if "messages" in agent_state.state and agent_state.state["messages"] is not None:
            if not isinstance(agent_state.state["messages"], list):
                raise ValueError(f"'messages' in AgentState was bad type: {type(agent_state.state['messages'])}")
            self._messages = agent_state.state["messages"]
        else:
            self._messages = initialize_message_sequence(
                self.model,
                self.system,
                self.memory,
            )

        # Interface must implement:
        # - internal_monologue
        # - assistant_message
        # - function_message
        # ...
        # Different interfaces can handle events differently
        # e.g., print in CLI vs send a discord message with a discord bot
        self.interface = interface

        # Create the persistence manager object based on the AgentState info
        # TODO
        self.persistence_manager = LocalStateManager(agent_state=agent_state)

        # Keep track of the total number of messages throughout all time
        self.messages_total = messages_total if messages_total is not None else (len(self._messages) - 1)  # (-system)
        # self.messages_total_init = self.messages_total
        self.messages_total_init = len(self._messages) - 1
        printd(f"Agent initialized, self.messages_total={self.messages_total}")

        # State needed for heartbeat pausing
        self.pause_heartbeats_start = None
        self.pause_heartbeats_minutes = 0

        self.first_message_verify_mono = first_message_verify_mono

        # Controls if the convo memory pressure warning is triggered
        # When an alert is sent in the message queue, set this to True (to avoid repeat alerts)
        # When the summarizer is run, set this back to False (to reset)
        self.agent_alerted_about_memory_pressure = False

        # Initialize the connection to the DB
        self.memgpt_config = MemGPTConfig()
        self.ms = MetadataStore(self.memgpt_config)

        # Create the agent in the DB
        self.save()

    @property
    def messages(self):
        return self._messages

    @messages.setter
    def messages(self, value):
        raise Exception("Modifying message list directly not allowed")

    def _trim_messages(self, num):
        """Trim messages from the front, not including the system message"""
        self.persistence_manager.trim_messages(num)

        new_messages = [self.messages[0]] + self.messages[num:]
        self._messages = new_messages

    def _prepend_to_messages(self, added_messages):
        """Wrapper around self.messages.prepend to allow additional calls to a state/persistence manager"""
        self.persistence_manager.prepend_to_messages(added_messages)

        new_messages = [self.messages[0]] + added_messages + self.messages[1:]  # prepend (no system)
        self._messages = new_messages
        self.messages_total += len(added_messages)  # still should increment the message counter (summaries are additions too)

    def _append_to_messages(self, added_messages):
        """Wrapper around self.messages.append to allow additional calls to a state/persistence manager"""
        self.persistence_manager.append_to_messages(added_messages)

        # strip extra metadata if it exists
        for msg in added_messages:
            msg.pop("api_response", None)
            msg.pop("api_args", None)
        new_messages = self.messages + added_messages  # append

        self._messages = new_messages
        self.messages_total += len(added_messages)

    def _swap_system_message(self, new_system_message):
        assert new_system_message["role"] == "system", new_system_message
        assert self.messages[0]["role"] == "system", self.messages
        self.persistence_manager.swap_system_message(new_system_message)

        new_messages = [new_system_message] + self.messages[1:]  # swap index 0 (system)
        self._messages = new_messages

    def _get_ai_reply(
        self,
        message_sequence,
        function_call="auto",
        first_message=False,  # hint
    ):
        """Get response from LLM API"""
        try:
            response = create(
                agent_config=self.config,
                messages=message_sequence,
                functions=self.functions,
                function_call=function_call,
                # hint
                first_message=first_message,
            )
            # special case for 'length'
            if response.choices[0].finish_reason == "length":
                raise Exception("Finish reason was length (maximum context length)")

            # catches for soft errors
            if response.choices[0].finish_reason not in ["stop", "function_call"]:
                raise Exception(f"API call finish with bad finish reason: {response}")

            # unpack with response.choices[0].message.content
            return response
        except Exception as e:
            raise e

    def _handle_ai_response(self, response_message):
        """Handles parsing and function execution"""
        messages = []  # append these to the history when done

        # Step 2: check if LLM wanted to call a function
        if response_message.get("function_call"):
            # The content if then internal monologue, not chat
            self.interface.internal_monologue(response_message.content)

            # generate UUID for tool call
            tool_call_id = str(uuid.uuid4())  # needs to be a string for JSON
            response_message["tool_call_id"] = tool_call_id
            # role: assistant (requesting tool call, set tool call ID)
            messages.append(response_message)  # extend conversation with assistant's reply
            printd(f"Function call message: {messages[-1]}")

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors

            # Failure case 1: function name is wrong
            function_name = response_message["function_call"]["name"]
            printd(f"Request to call function {function_name} with tool_call_id: {tool_call_id}")
            try:
                function_to_call = self.functions_python[function_name]
            except KeyError as e:
                error_msg = f"No function named {function_name}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    {"role": "function", "name": function_name, "content": function_response, "tool_call_id": tool_call_id}
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}")
                return messages, None, True  # force a heartbeat to allow agent to handle error

            # Failure case 2: function name is OK, but function args are bad JSON
            try:
                raw_function_args = response_message["function_call"]["arguments"]
                function_args = parse_json(raw_function_args)
            except Exception as e:
                error_msg = f"Error parsing JSON for function '{function_name}' arguments: {raw_function_args}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}")
                return messages, None, True  # force a heartbeat to allow agent to handle error

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            heartbeat_request = function_args.pop("request_heartbeat", None)
            if not (isinstance(heartbeat_request, bool) or heartbeat_request is None):
                printd(
                    f"{CLI_WARNING_PREFIX}'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )
                heartbeat_request = None

            # Failure case 3: function failed during execution
            self.interface.function_message(f"Running {function_name}({function_args})")
            try:
                function_args["self"] = self  # need to attach self to arg since it's dynamically linked
                function_response = function_to_call(**function_args)
                if function_name in ["conversation_search", "conversation_search_date", "archival_memory_search"]:
                    # with certain functions we rely on the paging mechanism to handle overflow
                    truncate = False
                else:
                    # but by default, we add a truncation safeguard to prevent bad functions from
                    # overflow the agent context window
                    truncate = True
                function_response_string = validate_function_response(function_response, truncate=truncate)
                function_args.pop("self", None)
                function_response = package_function_response(True, function_response_string)
                function_failed = False
            except Exception as e:
                function_args.pop("self", None)
                # error_msg = f"Error calling function {function_name} with args {function_args}: {str(e)}"
                # Less detailed - don't provide full args, idea is that it should be in recent context so no need (just adds noise)
                error_msg = f"Error calling function {function_name}: {str(e)}"
                error_msg_user = f"{error_msg}\n{traceback.format_exc()}"
                printd(error_msg_user)
                function_response = package_function_response(False, error_msg)
                messages.append(
                    {"role": "function", "name": function_name, "content": function_response, "tool_call_id": tool_call_id}
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}")
                return messages, None, True  # force a heartbeat to allow agent to handle error

            # If no failures happened along the way: ...
            # Step 4: send the info on the function call and function response to GPT
            self.interface.function_message(f"Success: {function_response_string}")
            messages.append(
                {"role": "function", "name": function_name, "content": function_response, "tool_call_id": tool_call_id}
            )  # extend conversation with function response

        else:
            # Standard non-function reply
            self.interface.internal_monologue(response_message.content)
            messages.append(response_message)  # extend conversation with assistant's reply
            heartbeat_request = None
            function_failed = None

        return messages, heartbeat_request, function_failed

    def step(self, user_message, first_message=False, first_message_retry_limit=FIRST_MESSAGE_ATTEMPTS, skip_verify=False):
        """Top-level event message handler for the MemGPT agent"""

        try:
            # Step 0: add user message
            if user_message is not None:
                self.interface.user_message(user_message)
                packed_user_message = {"role": "user", "content": user_message}
                # Special handling for AutoGen messages with 'name' field
                try:
                    user_message_json = json.loads(user_message)
                    # Treat 'name' as a special field
                    # If it exists in the input message, elevate it to the 'message' level
                    if "name" in user_message_json:
                        packed_user_message["name"] = user_message_json["name"]
                        user_message_json.pop("name", None)
                        packed_user_message["content"] = json.dumps(user_message_json, ensure_ascii=False)
                except Exception as e:
                    print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed with: {e}")
                input_message_sequence = self.messages + [packed_user_message]
            else:
                input_message_sequence = self.messages

            if len(input_message_sequence) > 1 and input_message_sequence[-1]["role"] != "user":
                printd(f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue")

            # Step 1: send the conversation and available functions to GPT
            if not skip_verify and (first_message or self.messages_total == self.messages_total_init):
                printd(f"This is the first message. Running extra verifier on AI response.")
                counter = 0
                while True:
                    response = self._get_ai_reply(
                        message_sequence=input_message_sequence,
                        first_message=True,  # passed through to the prompt formatter
                    )
                    if verify_first_message_correctness(response, require_monologue=self.first_message_verify_mono):
                        break

                    counter += 1
                    if counter > first_message_retry_limit:
                        raise Exception(f"Hit first message retry limit ({first_message_retry_limit})")

            else:
                response = self._get_ai_reply(
                    message_sequence=input_message_sequence,
                )

            # Step 2: check if LLM wanted to call a function
            # (if yes) Step 3: call the function
            # (if yes) Step 4: send the info on the function call and function response to LLM
            response_message = response.choices[0].message
            response_message_copy = response_message.copy()
            all_response_messages, heartbeat_request, function_failed = self._handle_ai_response(response_message)

            # Add the extra metadata to the assistant response
            # (e.g. enough metadata to enable recreating the API call)
            assert "api_response" not in all_response_messages[0]
            all_response_messages[0]["api_response"] = response_message_copy
            assert "api_args" not in all_response_messages[0]
            all_response_messages[0]["api_args"] = {
                "model": self.model,
                "messages": input_message_sequence,
                "functions": self.functions,
            }

            # Step 4: extend the message history
            if user_message is not None:
                all_new_messages = [packed_user_message] + all_response_messages
            else:
                all_new_messages = all_response_messages

            # Check the memory pressure and potentially issue a memory pressure warning
            current_total_tokens = response["usage"]["total_tokens"]
            active_memory_warning = False
            # We can't do summarize logic properly if context_window is undefined
            if self.config.llm_config.context_window is None:
                # Fallback if for some reason context_window is missing, just set to the default
                print(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
                print(f"{self.config}")
                self.config.llm_config.context_window = (
                    str(LLM_MAX_TOKENS[self.model])
                    if (self.model is not None and self.model in LLM_MAX_TOKENS)
                    else str(LLM_MAX_TOKENS["DEFAULT"])
                )
            if current_total_tokens > MESSAGE_SUMMARY_WARNING_FRAC * int(self.config.llm_config.context_window):
                printd(
                    f"{CLI_WARNING_PREFIX}last response total_tokens ({current_total_tokens}) > {MESSAGE_SUMMARY_WARNING_FRAC * int(self.config.llm_config.context_window)}"
                )
                # Only deliver the alert if we haven't already (this period)
                if not self.agent_alerted_about_memory_pressure:
                    active_memory_warning = True
                    self.agent_alerted_about_memory_pressure = True  # it's up to the outer loop to handle this
            else:
                printd(
                    f"last response total_tokens ({current_total_tokens}) < {MESSAGE_SUMMARY_WARNING_FRAC * int(self.config.llm_config.context_window)}"
                )

            self._append_to_messages(all_new_messages)
            return all_new_messages, heartbeat_request, function_failed, active_memory_warning

        except Exception as e:
            printd(f"step() failed\nuser_message = {user_message}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                # A separate API call to run a summarizer
                self.summarize_messages_inplace()

                # Try step again
                return self.step(user_message, first_message=first_message)
            else:
                printd(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e

    def summarize_messages_inplace(self, cutoff=None, preserve_last_N_messages=True):
        assert self.messages[0]["role"] == "system", f"self.messages[0] should be system (instead got {self.messages[0]})"

        # Start at index 1 (past the system message),
        # and collect messages for summarization until we reach the desired truncation token fraction (eg 50%)
        # Do not allow truncation of the last N messages, since these are needed for in-context examples of function calling
        token_counts = [count_tokens(str(msg)) for msg in self.messages]
        message_buffer_token_count = sum(token_counts[1:])  # no system message
        desired_token_count_to_summarize = int(message_buffer_token_count * MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC)
        candidate_messages_to_summarize = self.messages[1:]
        token_counts = token_counts[1:]
        if preserve_last_N_messages:
            candidate_messages_to_summarize = candidate_messages_to_summarize[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]
            token_counts = token_counts[:-MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST]
        printd(f"MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC={MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC}")
        printd(f"MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}")
        printd(f"token_counts={token_counts}")
        printd(f"message_buffer_token_count={message_buffer_token_count}")
        printd(f"desired_token_count_to_summarize={desired_token_count_to_summarize}")
        printd(f"len(candidate_messages_to_summarize)={len(candidate_messages_to_summarize)}")

        # If at this point there's nothing to summarize, throw an error
        if len(candidate_messages_to_summarize) == 0:
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(self.messages)}, preserve_N={MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST}]"
            )

        # Walk down the message buffer (front-to-back) until we hit the target token count
        tokens_so_far = 0
        cutoff = 0
        for i, msg in enumerate(candidate_messages_to_summarize):
            cutoff = i
            tokens_so_far += token_counts[i]
            if tokens_so_far > desired_token_count_to_summarize:
                break
        # Account for system message
        cutoff += 1

        # Try to make an assistant message come after the cutoff
        try:
            printd(f"Selected cutoff {cutoff} was a 'user', shifting one...")
            if self.messages[cutoff]["role"] == "user":
                new_cutoff = cutoff + 1
                if self.messages[new_cutoff]["role"] == "user":
                    printd(f"Shifted cutoff {new_cutoff} is still a 'user', ignoring...")
                cutoff = new_cutoff
        except IndexError:
            pass

        message_sequence_to_summarize = self.messages[1:cutoff]  # do NOT get rid of the system message
        if len(message_sequence_to_summarize) == 1:
            # This prevents a potential infinite loop of summarizing the same message over and over
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(message_sequence_to_summarize)} <= 1]"
            )
        else:
            printd(f"Attempting to summarize {len(message_sequence_to_summarize)} messages [1:{cutoff}] of {len(self.messages)}")

        # We can't do summarize logic properly if context_window is undefined
        if self.config.llm_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            print(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
            print(f"{self.config}")
            self.config.llm_config.context_window = (
                str(LLM_MAX_TOKENS[self.model])
                if (self.model is not None and self.model in LLM_MAX_TOKENS)
                else str(LLM_MAX_TOKENS["DEFAULT"])
            )
        summary = summarize_messages(agent_config=self.config, message_sequence_to_summarize=message_sequence_to_summarize)
        printd(f"Got summary: {summary}")

        # Metadata that's useful for the agent to see
        all_time_message_count = self.messages_total
        remaining_message_count = len(self.messages[cutoff:])
        hidden_message_count = all_time_message_count - remaining_message_count
        summary_message_count = len(message_sequence_to_summarize)
        summary_message = package_summarize_message(summary, summary_message_count, hidden_message_count, all_time_message_count)
        printd(f"Packaged into message: {summary_message}")

        prior_len = len(self.messages)
        self._trim_messages(cutoff)
        packed_summary_message = {"role": "user", "content": summary_message}
        self._prepend_to_messages([packed_summary_message])

        # reset alert
        self.agent_alerted_about_memory_pressure = False

        printd(f"Ran summarizer, messages length {prior_len} -> {len(self.messages)}")

    def heartbeat_is_paused(self):
        """Check if there's a requested pause on timed heartbeats"""

        # Check if the pause has been initiated
        if self.pause_heartbeats_start is None:
            return False

        # Check if it's been more than pause_heartbeats_minutes since pause_heartbeats_start
        elapsed_time = datetime.datetime.now() - self.pause_heartbeats_start
        return elapsed_time.total_seconds() < self.pause_heartbeats_minutes * 60

    def rebuild_memory(self):
        """Rebuilds the system message with the latest memory object"""
        curr_system_message = self.messages[0]  # this is the system + memory bank, not just the system prompt
        new_system_message = initialize_message_sequence(
            self.model,
            self.system,
            self.memory,
            archival_memory=self.persistence_manager.archival_memory,
            recall_memory=self.persistence_manager.recall_memory,
        )[0]

        diff = united_diff(curr_system_message["content"], new_system_message["content"])
        printd(f"Rebuilding system with new memory...\nDiff:\n{diff}")

        # Swap the system message out
        self._swap_system_message(new_system_message)

    def to_agent_state(self):
        # The state may have change since the last time we wrote it
        updated_state = {
            "persona": self.memory.persona,
            "human": self.memory.human,
            "system": self.system,
            "functions": self.functions,
            "messages": self.messages,
        }

        agent_state = AgentState(
            name=self.config.name,
            user_id=self.config.user_id,
            persona=self.config.persona,
            human=self.config.human,
            llm_config=self.config.llm_config,
            embedding_config=self.config.embedding_config,
            preset=self.config.preset,
            id=self.config.id,
            created_at=self.config.created_at,
            state=updated_state,
        )

        return agent_state

    def save(self):
        """Save agent state locally"""

        agent_state = self.to_agent_state()
        # TODO(swooders) does this make sense?
        # without this, even after Agent.__init__, agent.config.state["messages"] will be None
        self.config = agent_state

        # Check if we need to create the agent
        if not self.ms.get_agent(agent_id=agent_state.id, user_id=agent_state.user_id, agent_name=agent_state.name):
            self.ms.create_agent(agent=agent_state)
        else:
            # Otherwise, we should update the agent
            self.ms.update_agent(agent=agent_state)

        # # save config
        # self.config.save()

        # # save agent state to timestamped file
        # timestamp = get_local_time().replace(" ", "_").replace(":", "_")
        # filename = f"{timestamp}.json"
        # os.makedirs(self.config.save_state_dir(), exist_ok=True)
        # self.save_agent_state_json(os.path.join(self.config.save_state_dir(), filename))

        # # save the persistence manager too (recall/archival memory)
        # self.persistence_manager.save()

    ### Local state management
    # def to_dict(self):
    #     # TODO: select specific variables for the saves state (to eventually move to a DB) rather than checkpointing everything in the class
    #     return {
    #         "model": self.model,
    #         "system": self.system,
    #         "functions": self.functions,
    #         "messages": self.messages,  # TODO: convert to IDs
    #         "messages_total": self.messages_total,
    #         "memory": self.memory.to_dict(),
    #     }

    # def save_agent_state_json(self, filename):
    #     """Save agent state to JSON"""
    #     with open(filename, "w") as file:
    #         json.dump(self.to_dict(), file)

    # @classmethod
    # def load_agent(cls, interface, agent_config: AgentConfig):
    #     """Load saved agent state based on agent_config"""
    #     # TODO: support loading from specific file
    #     agent_name = agent_config.name

    #     # TODO: update this for metadata database

    #     # load state
    #     directory = agent_config.save_state_dir()
    #     json_files = glob.glob(os.path.join(directory, "*.json"))  # This will list all .json files in the current directory.
    #     if not json_files:
    #         print(f"/load error: no .json checkpoint files found")
    #         raise ValueError(f"Cannot load {agent_name} - no saved checkpoints found in {directory}")

    #     # Sort files based on modified timestamp, with the latest file being the first.
    #     filename = max(json_files, key=os.path.getmtime)
    #     state = json.load(open(filename, "r"))

    #     # load persistence manager
    #     persistence_manager = LocalStateManager.load(agent_config)

    #     messages = state["messages"]  # TODO: reconstruct messages using recall memory + stored IDs
    #     agent = cls(
    #         config=agent_config,
    #         model=state["model"],
    #         system=state["system"],
    #         functions=link_functions(state["functions"]),
    #         interface=interface,
    #         persistence_manager=persistence_manager,
    #         persistence_manager_init=False,
    #         persona_notes=state["memory"]["persona"],
    #         human_notes=state["memory"]["human"],
    #         messages_total=state["messages_total"] if "messages_total" in state else len(messages) - 1,
    #     )
    #     agent._messages = messages
    #     agent.memory = initialize_memory(state["memory"]["persona"], state["memory"]["human"])

    #     return agent
