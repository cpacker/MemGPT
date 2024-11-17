import datetime
import inspect
import traceback
import warnings
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

from letta.agent_store.storage import StorageConnector
from letta.constants import (
    CLI_WARNING_PREFIX,
    FIRST_MESSAGE_ATTEMPTS,
    FUNC_FAILED_HEARTBEAT_MESSAGE,
    IN_CONTEXT_MEMORY_KEYWORD,
    LLM_MAX_TOKENS,
    MESSAGE_SUMMARY_TRUNC_KEEP_N_LAST,
    MESSAGE_SUMMARY_TRUNC_TOKEN_FRAC,
    MESSAGE_SUMMARY_WARNING_FRAC,
    REQ_HEARTBEAT_MESSAGE,
)
from letta.errors import LLMError
from letta.helpers import ToolRulesSolver
from letta.interface import AgentInterface
from letta.llm_api.helpers import is_context_overflow_error
from letta.llm_api.llm_api_tools import create
from letta.local_llm.utils import num_tokens_from_functions, num_tokens_from_messages
from letta.memory import ArchivalMemory, RecallMemory, summarize_messages
from letta.metadata import MetadataStore
from letta.persistence_manager import LocalStateManager
from letta.schemas.agent import AgentState, AgentStepResponse
from letta.schemas.block import Block
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import MessageRole
from letta.schemas.memory import ContextWindowOverview, Memory
from letta.schemas.message import Message, UpdateMessage
from letta.schemas.openai.chat_completion_request import (
    Tool as ChatCompletionRequestTool,
)
from letta.schemas.openai.chat_completion_response import ChatCompletionResponse
from letta.schemas.openai.chat_completion_response import (
    Message as ChatCompletionMessage,
)
from letta.schemas.openai.chat_completion_response import UsageStatistics
from letta.schemas.passage import Passage
from letta.schemas.tool import Tool
from letta.schemas.tool_rule import TerminalToolRule
from letta.schemas.usage import LettaUsageStatistics
from letta.services.source_manager import SourceManager
from letta.services.user_manager import UserManager
from letta.streaming_interface import StreamingRefreshCLIInterface
from letta.system import (
    get_heartbeat,
    get_initial_boot_messages,
    get_login_event,
    get_token_limit_warning,
    package_function_response,
    package_summarize_message,
    package_user_message,
)
from letta.utils import (
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


def compile_memory_metadata_block(
    memory_edit_timestamp: datetime.datetime,
    archival_memory: Optional[ArchivalMemory] = None,
    recall_memory: Optional[RecallMemory] = None,
) -> str:
    # Put the timestamp in the local timezone (mimicking get_local_time())
    timestamp_str = memory_edit_timestamp.astimezone().strftime("%Y-%m-%d %I:%M:%S %p %Z%z").strip()

    # Create a metadata block of info so the agent knows about the metadata of out-of-context memories
    memory_metadata_block = "\n".join(
        [
            f"### Memory [last modified: {timestamp_str}]",
            f"{recall_memory.count() if recall_memory else 0} previous messages between you and the user are stored in recall memory (use functions to access them)",
            f"{archival_memory.count() if archival_memory else 0} total memories you created are stored in archival memory (use functions to access them)",
            "\nCore memory shown below (limited in size, additional information stored in archival / recall memory):",
        ]
    )
    return memory_metadata_block


def compile_system_message(
    system_prompt: str,
    in_context_memory: Memory,
    in_context_memory_last_edit: datetime.datetime,  # TODO move this inside of BaseMemory?
    archival_memory: Optional[ArchivalMemory] = None,
    recall_memory: Optional[RecallMemory] = None,
    user_defined_variables: Optional[dict] = None,
    append_icm_if_missing: bool = True,
    template_format: Literal["f-string", "mustache", "jinja2"] = "f-string",
) -> str:
    """Prepare the final/full system message that will be fed into the LLM API

    The base system message may be templated, in which case we need to render the variables.

    The following are reserved variables:
      - CORE_MEMORY: the in-context memory of the LLM
    """

    if user_defined_variables is not None:
        # TODO eventually support the user defining their own variables to inject
        raise NotImplementedError
    else:
        variables = {}

    # Add the protected memory variable
    if IN_CONTEXT_MEMORY_KEYWORD in variables:
        raise ValueError(f"Found protected variable '{IN_CONTEXT_MEMORY_KEYWORD}' in user-defined vars: {str(user_defined_variables)}")
    else:
        # TODO should this all put into the memory.__repr__ function?
        memory_metadata_string = compile_memory_metadata_block(
            memory_edit_timestamp=in_context_memory_last_edit,
            archival_memory=archival_memory,
            recall_memory=recall_memory,
        )
        full_memory_string = memory_metadata_string + "\n" + in_context_memory.compile()

        # Add to the variables list to inject
        variables[IN_CONTEXT_MEMORY_KEYWORD] = full_memory_string

    if template_format == "f-string":

        # Catch the special case where the system prompt is unformatted
        if append_icm_if_missing:
            memory_variable_string = "{" + IN_CONTEXT_MEMORY_KEYWORD + "}"
            if memory_variable_string not in system_prompt:
                # In this case, append it to the end to make sure memory is still injected
                # warnings.warn(f"{IN_CONTEXT_MEMORY_KEYWORD} variable was missing from system prompt, appending instead")
                system_prompt += "\n" + memory_variable_string

        # render the variables using the built-in templater
        try:
            formatted_prompt = system_prompt.format_map(variables)
        except Exception as e:
            raise ValueError(f"Failed to format system prompt - {str(e)}. System prompt value:\n{system_prompt}")

    else:
        # TODO support for mustache and jinja2
        raise NotImplementedError(template_format)

    return formatted_prompt


def initialize_message_sequence(
    model: str,
    system: str,
    memory: Memory,
    archival_memory: Optional[ArchivalMemory] = None,
    recall_memory: Optional[RecallMemory] = None,
    memory_edit_timestamp: Optional[datetime.datetime] = None,
    include_initial_boot_message: bool = True,
) -> List[dict]:
    if memory_edit_timestamp is None:
        memory_edit_timestamp = get_local_time()

    # full_system_message = construct_system_with_memory(
    # system, memory, memory_edit_timestamp, archival_memory=archival_memory, recall_memory=recall_memory
    # )
    full_system_message = compile_system_message(
        system_prompt=system,
        in_context_memory=memory,
        in_context_memory_last_edit=memory_edit_timestamp,
        archival_memory=archival_memory,
        recall_memory=recall_memory,
        user_defined_variables=None,
        append_icm_if_missing=True,
    )
    first_user_message = get_login_event()  # event letting Letta know the user just logged in

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


class BaseAgent(ABC):
    """
    Abstract class for all agents.
    Only two interfaces are required: step and update_state.
    """

    @abstractmethod
    def step(
        self,
        messages: Union[Message, List[Message]],
    ) -> LettaUsageStatistics:
        """
        Top-level event message handler for the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def update_state(self) -> AgentState:
        raise NotImplementedError


class Agent(BaseAgent):
    def __init__(
        self,
        interface: Optional[Union[AgentInterface, StreamingRefreshCLIInterface]],
        # agents can be created from providing agent_state
        agent_state: AgentState,
        tools: List[Tool],
        # memory: Memory,
        # extras
        messages_total: Optional[int] = None,  # TODO remove?
        first_message_verify_mono: bool = True,  # TODO move to config?
        initial_message_sequence: Optional[List[Message]] = None,
    ):
        assert isinstance(agent_state.memory, Memory), f"Memory object is not of type Memory: {type(agent_state.memory)}"
        # Hold a copy of the state that was used to init the agent
        self.agent_state = agent_state
        assert isinstance(self.agent_state.memory, Memory), f"Memory object is not of type Memory: {type(self.agent_state.memory)}"

        # link tools
        self.link_tools(tools)

        # initialize a tool rules solver
        if agent_state.tool_rules:
            # if there are tool rules, print out a warning
            for rule in agent_state.tool_rules:
                if not isinstance(rule, TerminalToolRule):
                    warnings.warn("Tool rules only work reliably for the latest OpenAI models that support structured outputs.")
                    break
        # add default rule for having send_message be a terminal tool
        if agent_state.tool_rules is None:
            agent_state.tool_rules = []
        # Define the rule to add
        send_message_terminal_rule = TerminalToolRule(tool_name="send_message")
        # Check if an equivalent rule is already present
        if not any(
            isinstance(rule, TerminalToolRule) and rule.tool_name == send_message_terminal_rule.tool_name for rule in agent_state.tool_rules
        ):
            agent_state.tool_rules.append(send_message_terminal_rule)

        self.tool_rules_solver = ToolRulesSolver(tool_rules=agent_state.tool_rules)

        # gpt-4, gpt-3.5-turbo, ...
        self.model = self.agent_state.llm_config.model

        # Store the system instructions (used to rebuild memory)
        self.system = self.agent_state.system

        # Initialize the memory object
        self.memory = self.agent_state.memory
        assert isinstance(self.memory, Memory), f"Memory object is not of type Memory: {type(self.memory)}"
        printd("Initialized memory object", self.memory.compile())

        # Interface must implement:
        # - internal_monologue
        # - assistant_message
        # - function_message
        # ...
        # Different interfaces can handle events differently
        # e.g., print in CLI vs send a discord message with a discord bot
        self.interface = interface

        # Create the persistence manager object based on the AgentState info
        self.persistence_manager = LocalStateManager(agent_state=self.agent_state)

        # State needed for heartbeat pausing
        self.pause_heartbeats_start = None
        self.pause_heartbeats_minutes = 0

        self.first_message_verify_mono = first_message_verify_mono

        # Controls if the convo memory pressure warning is triggered
        # When an alert is sent in the message queue, set this to True (to avoid repeat alerts)
        # When the summarizer is run, set this back to False (to reset)
        self.agent_alerted_about_memory_pressure = False

        self._messages: List[Message] = []

        # Once the memory object is initialized, use it to "bake" the system message
        if self.agent_state.message_ids is not None:
            self.set_message_buffer(message_ids=self.agent_state.message_ids)

        else:
            printd(f"Agent.__init__ :: creating, state={agent_state.message_ids}")
            assert self.agent_state.id is not None and self.agent_state.user_id is not None

            # Generate a sequence of initial messages to put in the buffer
            init_messages = initialize_message_sequence(
                model=self.model,
                system=self.system,
                memory=self.memory,
                archival_memory=None,
                recall_memory=None,
                memory_edit_timestamp=get_utc_time(),
                include_initial_boot_message=True,
            )

            if initial_message_sequence is not None:
                # We always need the system prompt up front
                system_message_obj = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=init_messages[0],
                )
                # Don't use anything else in the pregen sequence, instead use the provided sequence
                init_messages = [system_message_obj] + initial_message_sequence

            else:
                # Basic "more human than human" initial message sequence
                init_messages = initialize_message_sequence(
                    model=self.model,
                    system=self.system,
                    memory=self.memory,
                    archival_memory=None,
                    recall_memory=None,
                    memory_edit_timestamp=get_utc_time(),
                    include_initial_boot_message=True,
                )
                # Cast to Message objects
                init_messages = [
                    Message.dict_to_message(
                        agent_id=self.agent_state.id, user_id=self.agent_state.user_id, model=self.model, openai_message_dict=msg
                    )
                    for msg in init_messages
                ]

            # Cast the messages to actual Message objects to be synced to the DB
            init_messages_objs = []
            for msg in init_messages:
                init_messages_objs.append(msg)
            assert all([isinstance(msg, Message) for msg in init_messages_objs]), (init_messages_objs, init_messages)

            # Put the messages inside the message buffer
            self.messages_total = 0
            # self._append_to_messages(added_messages=[cast(Message, msg) for msg in init_messages_objs if msg is not None])
            self._append_to_messages(added_messages=init_messages_objs)
            self._validate_message_buffer_is_utc()

        # Keep track of the total number of messages throughout all time
        self.messages_total = messages_total if messages_total is not None else (len(self._messages) - 1)  # (-system)
        self.messages_total_init = len(self._messages) - 1
        printd(f"Agent initialized, self.messages_total={self.messages_total}")

        # Create the agent in the DB
        self.update_state()

    @property
    def messages(self) -> List[dict]:
        """Getter method that converts the internal Message list into OpenAI-style dicts"""
        return [msg.to_openai_dict() for msg in self._messages]

    @messages.setter
    def messages(self, value):
        raise Exception("Modifying message list directly not allowed")

    def link_tools(self, tools: List[Tool]):
        """Bind a tool object (schema + python function) to the agent object"""

        # tools
        for tool in tools:
            assert tool, f"Tool is None - must be error in querying tool from DB"
            assert tool.name in self.agent_state.tools, f"Tool {tool} not found in agent_state.tools"
        for tool_name in self.agent_state.tools:
            assert tool_name in [tool.name for tool in tools], f"Tool name {tool_name} not included in agent tool list"

        # Update tools
        self.tools = tools

        # Store the functions schemas (this is passed as an argument to ChatCompletion)
        self.functions = []
        self.functions_python = {}
        env = {}
        env.update(globals())
        for tool in tools:
            try:
                # WARNING: name may not be consistent?
                if tool.module:  # execute the whole module
                    exec(tool.module, env)
                else:
                    exec(tool.source_code, env)
                self.functions_python[tool.json_schema["name"]] = env[tool.json_schema["name"]]
                self.functions.append(tool.json_schema)
            except Exception as e:
                warnings.warn(f"WARNING: tool {tool.name} failed to link")
                print(e)
        assert all([callable(f) for k, f in self.functions_python.items()]), self.functions_python

    def _load_messages_from_recall(self, message_ids: List[str]) -> List[Message]:
        """Load a list of messages from recall storage"""

        # Pull the message objects from the database
        message_objs = []
        for msg_id in message_ids:
            msg_obj = self.persistence_manager.recall_memory.storage.get(msg_id)
            if msg_obj:
                if isinstance(msg_obj, Message):
                    message_objs.append(msg_obj)
                else:
                    printd(f"Warning - message ID {msg_id} is not a Message object")
                    warnings.warn(f"Warning - message ID {msg_id} is not a Message object")
            else:
                printd(f"Warning - message ID {msg_id} not found in recall storage")
                warnings.warn(f"Warning - message ID {msg_id} not found in recall storage")

        return message_objs

    def _validate_message_buffer_is_utc(self):
        """Iterate over the message buffer and force all messages to be UTC stamped"""

        for m in self._messages:
            # assert is_utc_datetime(m.created_at), f"created_at on message for agent {self.agent_state.name} isn't UTC:\n{vars(m)}"
            # TODO eventually do casting via an edit_message function
            if not is_utc_datetime(m.created_at):
                printd(f"Warning - created_at on message for agent {self.agent_state.name} isn't UTC (text='{m.text}')")
                m.created_at = m.created_at.replace(tzinfo=datetime.timezone.utc)

    def set_message_buffer(self, message_ids: List[str], force_utc: bool = True):
        """Set the messages in the buffer to the message IDs list"""

        message_objs = self._load_messages_from_recall(message_ids=message_ids)

        # set the objects in the buffer
        self._messages = message_objs

        # bugfix for old agents that may not have had UTC specified in their timestamps
        if force_utc:
            self._validate_message_buffer_is_utc()

        # also sync the message IDs attribute
        self.agent_state.message_ids = message_ids

    def refresh_message_buffer(self):
        """Refresh the message buffer from the database"""

        messages_to_sync = self.agent_state.message_ids
        assert messages_to_sync and all([isinstance(msg_id, str) for msg_id in messages_to_sync])

        self.set_message_buffer(message_ids=messages_to_sync)

    def _trim_messages(self, num):
        """Trim messages from the front, not including the system message"""
        self.persistence_manager.trim_messages(num)

        new_messages = [self._messages[0]] + self._messages[num:]
        self._messages = new_messages

    def _prepend_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.prepend to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.prepend_to_messages(added_messages)

        new_messages = [self._messages[0]] + added_messages + self._messages[1:]  # prepend (no system)
        self._messages = new_messages
        self.messages_total += len(added_messages)  # still should increment the message counter (summaries are additions too)

    def _append_to_messages(self, added_messages: List[Message]):
        """Wrapper around self.messages.append to allow additional calls to a state/persistence manager"""
        assert all([isinstance(msg, Message) for msg in added_messages])

        self.persistence_manager.append_to_messages(added_messages)

        # strip extra metadata if it exists
        # for msg in added_messages:
        # msg.pop("api_response", None)
        # msg.pop("api_args", None)
        new_messages = self._messages + added_messages  # append

        self._messages = new_messages
        self.messages_total += len(added_messages)

    def append_to_messages(self, added_messages: List[dict]):
        """An external-facing message append, where dict-like messages are first converted to Message objects"""
        added_messages_objs = [
            Message.dict_to_message(
                agent_id=self.agent_state.id,
                user_id=self.agent_state.user_id,
                model=self.model,
                openai_message_dict=msg,
            )
            for msg in added_messages
        ]
        self._append_to_messages(added_messages_objs)

    def _get_ai_reply(
        self,
        message_sequence: List[Message],
        function_call: str = "auto",
        first_message: bool = False,  # hint
        stream: bool = False,  # TODO move to config?
        fail_on_empty_response: bool = False,
        empty_response_retry_limit: int = 3,
    ) -> ChatCompletionResponse:
        """Get response from LLM API"""
        # Get the allowed tools based on the ToolRulesSolver state
        allowed_tool_names = self.tool_rules_solver.get_allowed_tool_names()

        if not allowed_tool_names:
            # if it's empty, any available tools are fair game
            allowed_functions = self.functions
        else:
            allowed_functions = [func for func in self.functions if func["name"] in allowed_tool_names]

        try:
            response = create(
                # agent_state=self.agent_state,
                llm_config=self.agent_state.llm_config,
                messages=message_sequence,
                user_id=self.agent_state.user_id,
                functions=allowed_functions,
                functions_python=self.functions_python,
                function_call=function_call,
                # hint
                first_message=first_message,
                # streaming
                stream=stream,
                stream_interface=self.interface,
            )

            if len(response.choices) == 0 or response.choices[0] is None:
                empty_api_err_message = f"API call didn't return a message: {response}"
                if fail_on_empty_response or empty_response_retry_limit == 0:
                    raise Exception(empty_api_err_message)
                else:
                    # Decrement retry limit and try again
                    warnings.warn(empty_api_err_message)
                    return self._get_ai_reply(
                        message_sequence, function_call, first_message, stream, fail_on_empty_response, empty_response_retry_limit - 1
                    )

            # special case for 'length'
            if response.choices[0].finish_reason == "length":
                raise Exception("Finish reason was length (maximum context length)")

            # catches for soft errors
            if response.choices[0].finish_reason not in ["stop", "function_call", "tool_calls"]:
                raise Exception(f"API call finish with bad finish reason: {response}")

            # unpack with response.choices[0].message.content
            return response
        except Exception as e:
            raise e

    def _handle_ai_response(
        self,
        response_message: ChatCompletionMessage,  # TODO should we eventually move the Message creation outside of this function?
        override_tool_call_id: bool = False,
        # If we are streaming, we needed to create a Message ID ahead of time,
        # and now we want to use it in the creation of the Message object
        # TODO figure out a cleaner way to do this
        response_message_id: Optional[str] = None,
    ) -> Tuple[List[Message], bool, bool]:
        """Handles parsing and function execution"""

        # Hacky failsafe for now to make sure we didn't implement the streaming Message ID creation incorrectly
        if response_message_id is not None:
            assert response_message_id.startswith("message-"), response_message_id

        messages = []  # append these to the history when done
        function_name = None

        # Step 2: check if LLM wanted to call a function
        if response_message.function_call or (response_message.tool_calls is not None and len(response_message.tool_calls) > 0):
            if response_message.function_call:
                raise DeprecationWarning(response_message)
            if response_message.tool_calls is not None and len(response_message.tool_calls) > 1:
                # raise NotImplementedError(f">1 tool call not supported")
                # TODO eventually support sequential tool calling
                printd(f">1 tool call not supported, using index=0 only\n{response_message.tool_calls}")
                response_message.tool_calls = [response_message.tool_calls[0]]
            assert response_message.tool_calls is not None and len(response_message.tool_calls) > 0

            # generate UUID for tool call
            if override_tool_call_id or response_message.function_call:
                warnings.warn("Overriding the tool call can result in inconsistent tool call IDs during streaming")
                tool_call_id = get_tool_call_id()  # needs to be a string for JSON
                response_message.tool_calls[0].id = tool_call_id
            else:
                tool_call_id = response_message.tool_calls[0].id
                assert tool_call_id is not None  # should be defined

            # only necessary to add the tool_cal_id to a function call (antipattern)
            # response_message_dict = response_message.model_dump()
            # response_message_dict["tool_call_id"] = tool_call_id

            # role: assistant (requesting tool call, set tool call ID)
            messages.append(
                # NOTE: we're recreating the message here
                # TODO should probably just overwrite the fields?
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            printd(f"Function call message: {messages[-1]}")

            nonnull_content = False
            if response_message.content:
                # The content if then internal monologue, not chat
                self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])
                # Flag to avoid printing a duplicate if inner thoughts get popped from the function call
                nonnull_content = True

            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            function_call = (
                response_message.function_call if response_message.function_call is not None else response_message.tool_calls[0].function
            )

            # Get the name of the function
            function_name = function_call.name
            printd(f"Request to call function {function_name} with tool_call_id: {tool_call_id}")

            # Failure case 1: function name is wrong
            try:
                function_to_call = self.functions_python[function_name]
            except KeyError:
                error_msg = f"No function named {function_name}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                return messages, False, True  # force a heartbeat to allow agent to handle error

            # Failure case 2: function name is OK, but function args are bad JSON
            try:
                raw_function_args = function_call.arguments
                function_args = parse_json(raw_function_args)
            except Exception:
                error_msg = f"Error parsing JSON for function '{function_name}' arguments: {function_call.arguments}"
                function_response = package_function_response(False, error_msg)
                messages.append(
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                return messages, False, True  # force a heartbeat to allow agent to handle error

            # Check if inner thoughts is in the function call arguments (possible apparently if you are using Azure)
            if "inner_thoughts" in function_args:
                response_message.content = function_args.pop("inner_thoughts")
            # The content if then internal monologue, not chat
            if response_message.content and not nonnull_content:
                self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])

            # (Still parsing function args)
            # Handle requests for immediate heartbeat
            heartbeat_request = function_args.pop("request_heartbeat", None)

            # Edge case: heartbeat_request is returned as a stringified boolean, we will attempt to parse:
            if isinstance(heartbeat_request, str) and heartbeat_request.lower().strip() == "true":
                heartbeat_request = True

            if not isinstance(heartbeat_request, bool) or heartbeat_request is None:
                printd(
                    f"{CLI_WARNING_PREFIX}'request_heartbeat' arg parsed was not a bool or None, type={type(heartbeat_request)}, value={heartbeat_request}"
                )
                heartbeat_request = False

            # Failure case 3: function failed during execution
            # NOTE: the msg_obj associated with the "Running " message is the prior assistant message, not the function/tool role message
            #       this is because the function/tool role message is only created once the function/tool has executed/returned
            self.interface.function_message(f"Running {function_name}({function_args})", msg_obj=messages[-1])
            try:
                spec = inspect.getfullargspec(function_to_call).annotations

                for name, arg in function_args.items():
                    if isinstance(function_args[name], dict):
                        function_args[name] = spec[name](**function_args[name])

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
                    Message.dict_to_message(
                        agent_id=self.agent_state.id,
                        user_id=self.agent_state.user_id,
                        model=self.model,
                        openai_message_dict={
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                            "tool_call_id": tool_call_id,
                        },
                    )
                )  # extend conversation with function response
                self.interface.function_message(f"Ran {function_name}({function_args})", msg_obj=messages[-1])
                self.interface.function_message(f"Error: {error_msg}", msg_obj=messages[-1])
                return messages, False, True  # force a heartbeat to allow agent to handle error

            # If no failures happened along the way: ...
            # Step 4: send the info on the function call and function response to GPT
            messages.append(
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                        "tool_call_id": tool_call_id,
                    },
                )
            )  # extend conversation with function response
            self.interface.function_message(f"Ran {function_name}({function_args})", msg_obj=messages[-1])
            self.interface.function_message(f"Success: {function_response_string}", msg_obj=messages[-1])

        else:
            # Standard non-function reply
            messages.append(
                Message.dict_to_message(
                    id=response_message_id,
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=response_message.model_dump(),
                )
            )  # extend conversation with assistant's reply
            self.interface.internal_monologue(response_message.content, msg_obj=messages[-1])
            heartbeat_request = False
            function_failed = False

        # rebuild memory
        # TODO: @charles please check this
        self.rebuild_memory()

        # Update ToolRulesSolver state with last called function
        self.tool_rules_solver.update_tool_usage(function_name)
        # Update heartbeat request according to provided tool rules
        if self.tool_rules_solver.has_children_tools(function_name):
            heartbeat_request = True
        elif self.tool_rules_solver.is_terminal_tool(function_name):
            heartbeat_request = False

        return messages, heartbeat_request, function_failed

    def step(
        self,
        messages: Union[Message, List[Message]],
        # additional args
        chaining: bool = True,
        max_chaining_steps: Optional[int] = None,
        ms: Optional[MetadataStore] = None,
        **kwargs,
    ) -> LettaUsageStatistics:
        """Run Agent.step in a loop, handling chaining via heartbeat requests and function failures"""
        # assert ms is not None, "MetadataStore is required"

        next_input_message = messages if isinstance(messages, list) else [messages]
        counter = 0
        total_usage = UsageStatistics()
        step_count = 0
        while True:
            kwargs["ms"] = ms
            kwargs["first_message"] = False
            step_response = self.inner_step(
                messages=next_input_message,
                **kwargs,
            )
            step_response.messages
            heartbeat_request = step_response.heartbeat_request
            function_failed = step_response.function_failed
            token_warning = step_response.in_context_memory_warning
            usage = step_response.usage

            step_count += 1
            total_usage += usage
            counter += 1
            self.interface.step_complete()

            # logger.debug("Saving agent state")
            # save updated state
            if ms:
                save_agent(self, ms)

            # Chain stops
            if not chaining:
                printd("No chaining, stopping after one step")
                break
            elif max_chaining_steps is not None and counter > max_chaining_steps:
                printd(f"Hit max chaining steps, stopping after {counter} steps")
                break
            # Chain handlers
            elif token_warning:
                assert self.agent_state.user_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_token_limit_warning(),
                    },
                )
                continue  # always chain
            elif function_failed:
                assert self.agent_state.user_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_heartbeat(FUNC_FAILED_HEARTBEAT_MESSAGE),
                    },
                )
                continue  # always chain
            elif heartbeat_request:
                assert self.agent_state.user_id is not None
                next_input_message = Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict={
                        "role": "user",  # TODO: change to system?
                        "content": get_heartbeat(REQ_HEARTBEAT_MESSAGE),
                    },
                )
                continue  # always chain
            # Letta no-op / yield
            else:
                break

        return LettaUsageStatistics(**total_usage.model_dump(), step_count=step_count)

    def inner_step(
        self,
        messages: Union[Message, List[Message]],
        first_message: bool = False,
        first_message_retry_limit: int = FIRST_MESSAGE_ATTEMPTS,
        skip_verify: bool = False,
        stream: bool = False,  # TODO move to config?
        ms: Optional[MetadataStore] = None,
    ) -> AgentStepResponse:
        """Runs a single step in the agent loop (generates at most one LLM call)"""

        try:

            # Step 0: update core memory
            # only pulling latest block data if shared memory is being used
            # TODO: ensure we're passing in metadata store from all surfaces
            if ms is not None:
                should_update = False
                for block in self.agent_state.memory.to_dict()["memory"].values():
                    if not block.get("template", False):
                        should_update = True
                if should_update:
                    # TODO: the force=True can be optimized away
                    # once we ensure we're correctly comparing whether in-memory core
                    # data is different than persisted core data.
                    self.rebuild_memory(force=True, ms=ms)

            # Step 1: add user message
            if isinstance(messages, Message):
                messages = [messages]

            if not all(isinstance(m, Message) for m in messages):
                raise ValueError(f"messages should be a Message or a list of Message, got {type(messages)}")

            input_message_sequence = self._messages + messages

            if len(input_message_sequence) > 1 and input_message_sequence[-1].role != "user":
                printd(f"{CLI_WARNING_PREFIX}Attempting to run ChatCompletion without user as the last message in the queue")

            # Step 2: send the conversation and available functions to the LLM
            if not skip_verify and (first_message or self.messages_total == self.messages_total_init):
                printd(f"This is the first message. Running extra verifier on AI response.")
                counter = 0
                while True:
                    response = self._get_ai_reply(
                        message_sequence=input_message_sequence, first_message=True, stream=stream  # passed through to the prompt formatter
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

            # Step 6: extend the message history
            if len(messages) > 0:
                all_new_messages = messages + all_response_messages
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

            # update state after each step
            self.update_state()

            return AgentStepResponse(
                messages=all_new_messages,
                heartbeat_request=heartbeat_request,
                function_failed=function_failed,
                in_context_memory_warning=active_memory_warning,
                usage=response.usage,
            )

        except Exception as e:
            printd(f"step() failed\nmessages = {messages}\nerror = {e}")

            # If we got a context alert, try trimming the messages length, then try again
            if is_context_overflow_error(e):
                # A separate API call to run a summarizer
                self.summarize_messages_inplace()

                # Try step again
                return self.inner_step(
                    messages=messages,
                    first_message=first_message,
                    first_message_retry_limit=first_message_retry_limit,
                    skip_verify=skip_verify,
                    stream=stream,
                    ms=ms,
                )

            else:
                printd(f"step() failed with an unrecognized exception: '{str(e)}'")
                raise e

    def step_user_message(self, user_message_str: str, **kwargs) -> AgentStepResponse:
        """Takes a basic user message string, turns it into a stringified JSON with extra metadata, then sends it to the agent

        Example:
        -> user_message_str = 'hi'
        -> {'message': 'hi', 'type': 'user_message', ...}
        -> json.dumps(...)
        -> agent.step(messages=[Message(role='user', text=...)])
        """
        # Wrap with metadata, dumps to JSON
        assert user_message_str and isinstance(
            user_message_str, str
        ), f"user_message_str should be a non-empty string, got {type(user_message_str)}"
        user_message_json_str = package_user_message(user_message_str)

        # Validate JSON via save/load
        user_message = validate_json(user_message_json_str)
        cleaned_user_message_text, name = strip_name_field_from_user_message(user_message)

        # Turn into a dict
        openai_message_dict = {"role": "user", "content": cleaned_user_message_text, "name": name}

        # Create the associated Message object (in the database)
        assert self.agent_state.user_id is not None, "User ID is not set"
        user_message = Message.dict_to_message(
            agent_id=self.agent_state.id,
            user_id=self.agent_state.user_id,
            model=self.model,
            openai_message_dict=openai_message_dict,
            # created_at=timestamp,
        )

        return self.inner_step(messages=[user_message], **kwargs)

    def summarize_messages_inplace(self, cutoff=None, preserve_last_N_messages=True, disallow_tool_as_first=True):
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

        # Make sure the cutoff isn't on a 'tool' or 'function'
        if disallow_tool_as_first:
            while self.messages[cutoff]["role"] in ["tool", "function"] and cutoff < len(self.messages):
                printd(f"Selected cutoff {cutoff} was a 'tool', shifting one...")
                cutoff += 1

        message_sequence_to_summarize = self._messages[1:cutoff]  # do NOT get rid of the system message
        if len(message_sequence_to_summarize) <= 1:
            # This prevents a potential infinite loop of summarizing the same message over and over
            raise LLMError(
                f"Summarize error: tried to run summarize, but couldn't find enough messages to compress [len={len(message_sequence_to_summarize)} <= 1]"
            )
        else:
            printd(f"Attempting to summarize {len(message_sequence_to_summarize)} messages [1:{cutoff}] of {len(self._messages)}")

        # We can't do summarize logic properly if context_window is undefined
        if self.agent_state.llm_config.context_window is None:
            # Fallback if for some reason context_window is missing, just set to the default
            print(f"{CLI_WARNING_PREFIX}could not find context_window in config, setting to default {LLM_MAX_TOKENS['DEFAULT']}")
            print(f"{self.agent_state}")
            self.agent_state.llm_config.context_window = (
                LLM_MAX_TOKENS[self.model] if (self.model is not None and self.model in LLM_MAX_TOKENS) else LLM_MAX_TOKENS["DEFAULT"]
            )
        summary = summarize_messages(agent_state=self.agent_state, message_sequence_to_summarize=message_sequence_to_summarize)
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
        self._prepend_to_messages(
            [
                Message.dict_to_message(
                    agent_id=self.agent_state.id,
                    user_id=self.agent_state.user_id,
                    model=self.model,
                    openai_message_dict=packed_summary_message,
                )
            ]
        )

        # reset alert
        self.agent_alerted_about_memory_pressure = False

        printd(f"Ran summarizer, messages length {prior_len} -> {len(self.messages)}")

    def heartbeat_is_paused(self):
        """Check if there's a requested pause on timed heartbeats"""

        # Check if the pause has been initiated
        if self.pause_heartbeats_start is None:
            return False

        # Check if it's been more than pause_heartbeats_minutes since pause_heartbeats_start
        elapsed_time = get_utc_time() - self.pause_heartbeats_start
        return elapsed_time.total_seconds() < self.pause_heartbeats_minutes * 60

    def _swap_system_message_in_buffer(self, new_system_message: str):
        """Update the system message (NOT prompt) of the Agent (requires updating the internal buffer)"""
        assert isinstance(new_system_message, str)
        new_system_message_obj = Message.dict_to_message(
            agent_id=self.agent_state.id,
            user_id=self.agent_state.user_id,
            model=self.model,
            openai_message_dict={"role": "system", "content": new_system_message},
        )

        assert new_system_message_obj.role == "system", new_system_message_obj
        assert self._messages[0].role == "system", self._messages

        self.persistence_manager.swap_system_message(new_system_message_obj)

        new_messages = [new_system_message_obj] + self._messages[1:]  # swap index 0 (system)
        self._messages = new_messages

    def rebuild_memory(self, force=False, update_timestamp=True, ms: Optional[MetadataStore] = None):
        """Rebuilds the system message with the latest memory object and any shared memory block updates"""
        curr_system_message = self.messages[0]  # this is the system + memory bank, not just the system prompt

        # NOTE: This is a hacky way to check if the memory has changed
        memory_repr = self.memory.compile()
        if not force and memory_repr == curr_system_message["content"][-(len(memory_repr)) :]:
            printd(f"Memory has not changed, not rebuilding system")
            return

        if ms:
            for block in self.memory.to_dict()["memory"].values():
                if block.get("templates", False):
                    # we don't expect to update shared memory blocks that
                    # are templates. this is something we could update in the
                    # future if we expect templates to change often.
                    continue
                block_id = block.get("id")
                db_block = ms.get_block(block_id=block_id)
                if db_block is None:
                    # this case covers if someone has deleted a shared block by interacting
                    # with some other agent.
                    # in that case we should remove this shared block from the agent currently being
                    # evaluated.
                    printd(f"removing block: {block_id=}")
                    continue
                if not isinstance(db_block.value, str):
                    printd(f"skipping block update, unexpected value: {block_id=}")
                    continue
                # TODO: we may want to update which columns we're updating from shared memory e.g. the limit
                self.memory.update_block_value(label=block.get("label", ""), value=db_block.value)

        # If the memory didn't update, we probably don't want to update the timestamp inside
        # For example, if we're doing a system prompt swap, this should probably be False
        if update_timestamp:
            memory_edit_timestamp = get_utc_time()
        else:
            # NOTE: a bit of a hack - we pull the timestamp from the message created_by
            memory_edit_timestamp = self._messages[0].created_at

        # update memory (TODO: potentially update recall/archival stats seperately)
        new_system_message_str = compile_system_message(
            system_prompt=self.system,
            in_context_memory=self.memory,
            in_context_memory_last_edit=memory_edit_timestamp,
            archival_memory=self.persistence_manager.archival_memory,
            recall_memory=self.persistence_manager.recall_memory,
            user_defined_variables=None,
            append_icm_if_missing=True,
        )
        new_system_message = {
            "role": "system",
            "content": new_system_message_str,
        }

        diff = united_diff(curr_system_message["content"], new_system_message["content"])
        if len(diff) > 0:  # there was a diff
            printd(f"Rebuilding system with new memory...\nDiff:\n{diff}")

            # Swap the system message out (only if there is a diff)
            self._swap_system_message_in_buffer(new_system_message=new_system_message_str)
            assert self.messages[0]["content"] == new_system_message["content"], (
                self.messages[0]["content"],
                new_system_message["content"],
            )

    def update_system_prompt(self, new_system_prompt: str):
        """Update the system prompt of the agent (requires rebuilding the memory block if there's a difference)"""
        assert isinstance(new_system_prompt, str)

        if new_system_prompt == self.system:
            input("same???")
            return

        self.system = new_system_prompt

        # updating the system prompt requires rebuilding the memory block inside the compiled system message
        self.rebuild_memory(force=True, update_timestamp=False)

        # make sure to persist the change
        _ = self.update_state()

    def add_function(self, function_name: str) -> str:
        # TODO: refactor
        raise NotImplementedError

    def remove_function(self, function_name: str) -> str:
        # TODO: refactor
        raise NotImplementedError

    def update_state(self) -> AgentState:
        message_ids = [msg.id for msg in self._messages]
        assert isinstance(self.memory, Memory), f"Memory is not a Memory object: {type(self.memory)}"

        # override any fields that may have been updated
        self.agent_state.message_ids = message_ids
        self.agent_state.memory = self.memory
        self.agent_state.system = self.system

        return self.agent_state

    def migrate_embedding(self, embedding_config: EmbeddingConfig):
        """Migrate the agent to a new embedding"""
        # TODO: archival memory

        # TODO: recall memory
        raise NotImplementedError()

    def attach_source(self, source_id: str, source_connector: StorageConnector, ms: MetadataStore):
        """Attach data with name `source_name` to the agent from source_connector."""
        # TODO: eventually, adding a data source should just give access to the retriever the source table, rather than modifying archival memory
        user = UserManager().get_user_by_id(self.agent_state.user_id)
        filters = {"user_id": self.agent_state.user_id, "source_id": source_id}
        size = source_connector.size(filters)
        page_size = 100
        generator = source_connector.get_all_paginated(filters=filters, page_size=page_size)  # yields List[Passage]
        all_passages = []
        for i in tqdm(range(0, size, page_size)):
            passages = next(generator)

            # need to associated passage with agent (for filtering)
            for passage in passages:
                assert isinstance(passage, Passage), f"Generate yielded bad non-Passage type: {type(passage)}"
                passage.agent_id = self.agent_state.id

                # regenerate passage ID (avoid duplicates)
                # TODO: need to find another solution to the text duplication issue
                # passage.id = create_uuid_from_string(f"{source_id}_{str(passage.agent_id)}_{passage.text}")

            # insert into agent archival memory
            self.persistence_manager.archival_memory.storage.insert_many(passages)
            all_passages += passages

        assert size == len(all_passages), f"Expected {size} passages, but only got {len(all_passages)}"

        # save destination storage
        self.persistence_manager.archival_memory.storage.save()

        # attach to agent
        source = SourceManager().get_source_by_id(source_id=source_id, actor=user)
        assert source is not None, f"Source {source_id} not found in metadata store"
        ms.attach_source(agent_id=self.agent_state.id, source_id=source_id, user_id=self.agent_state.user_id)

        total_agent_passages = self.persistence_manager.archival_memory.storage.size()

        printd(
            f"Attached data source {source.name} to agent {self.agent_state.name}, consisting of {len(all_passages)}. Agent now has {total_agent_passages} embeddings in archival memory.",
        )

    def update_message(self, request: UpdateMessage) -> Message:
        """Update the details of a message associated with an agent"""

        message = self.persistence_manager.recall_memory.storage.get(id=request.id)
        if message is None:
            raise ValueError(f"Message with id {request.id} not found")
        assert isinstance(message, Message), f"Message is not a Message object: {type(message)}"

        # Override fields
        # NOTE: we try to do some sanity checking here (see asserts), but it's not foolproof
        if request.role:
            message.role = request.role
        if request.text:
            message.text = request.text
        if request.name:
            message.name = request.name
        if request.tool_calls:
            assert message.role == MessageRole.assistant, "Tool calls can only be added to assistant messages"
            message.tool_calls = request.tool_calls
        if request.tool_call_id:
            assert message.role == MessageRole.tool, "tool_call_id can only be added to tool messages"
            message.tool_call_id = request.tool_call_id

        # Save the updated message
        self.persistence_manager.recall_memory.storage.update(record=message)

        # Return the updated message
        updated_message = self.persistence_manager.recall_memory.storage.get(id=message.id)
        if updated_message is None:
            raise ValueError(f"Error persisting message - message with id {request.id} not found")
        return updated_message

    # TODO(sarah): should we be creating a new message here, or just editing a message?
    def rethink_message(self, new_thought: str) -> Message:
        """Rethink / update the last message"""
        for x in range(len(self.messages) - 1, 0, -1):
            msg_obj = self._messages[x]
            if msg_obj.role == MessageRole.assistant:
                updated_message = self.update_message(
                    request=UpdateMessage(
                        id=msg_obj.id,
                        text=new_thought,
                    )
                )
                self.refresh_message_buffer()
                return updated_message
        raise ValueError(f"No assistant message found to update")

    # TODO(sarah): should we be creating a new message here, or just editing a message?
    def rewrite_message(self, new_text: str) -> Message:
        """Rewrite / update the send_message text on the last message"""

        # Walk backwards through the messages until we find an assistant message
        for x in range(len(self._messages) - 1, 0, -1):
            if self._messages[x].role == MessageRole.assistant:
                # Get the current message content
                message_obj = self._messages[x]

                # The rewrite target is the output of send_message
                if message_obj.tool_calls is not None and len(message_obj.tool_calls) > 0:

                    # Check that we hit an assistant send_message call
                    name_string = message_obj.tool_calls[0].function.name
                    if name_string is None or name_string != "send_message":
                        raise ValueError("Assistant missing send_message function call")

                    args_string = message_obj.tool_calls[0].function.arguments
                    if args_string is None:
                        raise ValueError("Assistant missing send_message function arguments")

                    args_json = json_loads(args_string)
                    if "message" not in args_json:
                        raise ValueError("Assistant missing send_message message argument")

                    # Once we found our target, rewrite it
                    args_json["message"] = new_text
                    new_args_string = json_dumps(args_json)
                    message_obj.tool_calls[0].function.arguments = new_args_string

                    # Write the update to the DB
                    updated_message = self.update_message(
                        request=UpdateMessage(
                            id=message_obj.id,
                            tool_calls=message_obj.tool_calls,
                        )
                    )
                    self.refresh_message_buffer()
                    return updated_message

        raise ValueError("No assistant message found to update")

    def pop_message(self, count: int = 1) -> List[Message]:
        """Pop the last N messages from the agent's memory"""
        n_messages = len(self._messages)
        popped_messages = []
        MIN_MESSAGES = 2
        if n_messages <= MIN_MESSAGES:
            raise ValueError(f"Agent only has {n_messages} messages in stack, none left to pop")
        elif n_messages - count < MIN_MESSAGES:
            raise ValueError(f"Agent only has {n_messages} messages in stack, cannot pop more than {n_messages - MIN_MESSAGES}")
        else:
            # print(f"Popping last {count} messages from stack")
            for _ in range(min(count, len(self._messages))):
                # remove the message from the internal state of the agent
                deleted_message = self._messages.pop()
                # then also remove it from recall storage
                try:
                    self.persistence_manager.recall_memory.storage.delete(filters={"id": deleted_message.id})
                    popped_messages.append(deleted_message)
                except Exception as e:
                    warnings.warn(f"Error deleting message {deleted_message.id} from recall memory: {e}")
                    self._messages.append(deleted_message)
                    break

        return popped_messages

    def pop_until_user(self) -> List[Message]:
        """Pop all messages until the last user message"""
        if MessageRole.user not in [msg.role for msg in self._messages]:
            raise ValueError("No user message found in buffer")

        popped_messages = []
        while len(self._messages) > 0:
            if self._messages[-1].role == MessageRole.user:
                # we want to pop up to the last user message
                return popped_messages
            else:
                popped_messages.append(self.pop_message(count=1))

        raise ValueError("No user message found in buffer")

    def retry_message(self) -> List[Message]:
        """Retry / regenerate the last message"""

        self.pop_until_user()
        user_message = self.pop_message(count=1)[0]
        assert user_message.text is not None, "User message text is None"
        step_response = self.step_user_message(user_message_str=user_message.text)
        messages = step_response.messages

        assert messages is not None
        assert all(isinstance(msg, Message) for msg in messages), "step() returned non-Message objects"
        return messages

    def get_context_window(self) -> ContextWindowOverview:
        """Get the context window of the agent"""

        system_prompt = self.agent_state.system  # TODO is this the current system or the initial system?
        num_tokens_system = count_tokens(system_prompt)
        core_memory = self.memory.compile()
        num_tokens_core_memory = count_tokens(core_memory)

        # conversion of messages to OpenAI dict format, which is passed to the token counter
        messages_openai_format = self.messages

        # Check if there's a summary message in the message queue
        if (
            len(self._messages) > 1
            and self._messages[1].role == MessageRole.user
            and isinstance(self._messages[1].text, str)
            # TODO remove hardcoding
            and "The following is a summary of the previous " in self._messages[1].text
        ):
            # Summary message exists
            assert self._messages[1].text is not None
            summary_memory = self._messages[1].text
            num_tokens_summary_memory = count_tokens(self._messages[1].text)
            # with a summary message, the real messages start at index 2
            num_tokens_messages = (
                num_tokens_from_messages(messages=messages_openai_format[2:], model=self.model) if len(messages_openai_format) > 2 else 0
            )

        else:
            summary_memory = None
            num_tokens_summary_memory = 0
            # with no summary message, the real messages start at index 1
            num_tokens_messages = (
                num_tokens_from_messages(messages=messages_openai_format[1:], model=self.model) if len(messages_openai_format) > 1 else 0
            )

        num_archival_memory = self.persistence_manager.archival_memory.storage.size()
        num_recall_memory = self.persistence_manager.recall_memory.storage.size()
        external_memory_summary = compile_memory_metadata_block(
            memory_edit_timestamp=get_utc_time(),  # dummy timestamp
            archival_memory=self.persistence_manager.archival_memory,
            recall_memory=self.persistence_manager.recall_memory,
        )
        num_tokens_external_memory_summary = count_tokens(external_memory_summary)

        # tokens taken up by function definitions
        if self.functions:
            available_functions_definitions = [ChatCompletionRequestTool(type="function", function=f) for f in self.functions]
            num_tokens_available_functions_definitions = num_tokens_from_functions(functions=self.functions, model=self.model)
        else:
            available_functions_definitions = []
            num_tokens_available_functions_definitions = 0

        num_tokens_used_total = (
            num_tokens_system  # system prompt
            + num_tokens_available_functions_definitions  # function definitions
            + num_tokens_core_memory  # core memory
            + num_tokens_external_memory_summary  # metadata (statistics) about recall/archival
            + num_tokens_summary_memory  # summary of ongoing conversation
            + num_tokens_messages  # tokens taken by messages
        )
        assert isinstance(num_tokens_used_total, int)

        return ContextWindowOverview(
            # context window breakdown (in messages)
            num_messages=len(self._messages),
            num_archival_memory=num_archival_memory,
            num_recall_memory=num_recall_memory,
            num_tokens_external_memory_summary=num_tokens_external_memory_summary,
            # top-level information
            context_window_size_max=self.agent_state.llm_config.context_window,
            context_window_size_current=num_tokens_used_total,
            # context window breakdown (in tokens)
            num_tokens_system=num_tokens_system,
            system_prompt=system_prompt,
            num_tokens_core_memory=num_tokens_core_memory,
            core_memory=core_memory,
            num_tokens_summary_memory=num_tokens_summary_memory,
            summary_memory=summary_memory,
            num_tokens_messages=num_tokens_messages,
            messages=self._messages,
            # related to functions
            num_tokens_functions_definitions=num_tokens_available_functions_definitions,
            functions_definitions=available_functions_definitions,
        )

    def count_tokens(self) -> int:
        """Count the tokens in the current context window"""
        return self.get_context_window().context_window_size_current


def save_agent(agent: Agent, ms: MetadataStore):
    """Save agent to metadata store"""

    agent.update_state()
    agent_state = agent.agent_state
    agent_id = agent_state.id
    assert isinstance(agent_state.memory, Memory), f"Memory is not a Memory object: {type(agent_state.memory)}"

    # NOTE: we're saving agent memory before persisting the agent to ensure
    # that allocated block_ids for each memory block are present in the agent model
    save_agent_memory(agent=agent, ms=ms)

    if ms.get_agent(agent_id=agent.agent_state.id):
        ms.update_agent(agent_state)
    else:
        ms.create_agent(agent_state)

    agent.agent_state = ms.get_agent(agent_id=agent_id)
    assert isinstance(agent.agent_state.memory, Memory), f"Memory is not a Memory object: {type(agent_state.memory)}"


def save_agent_memory(agent: Agent, ms: MetadataStore):
    """
    Save agent memory to metadata store. Memory is a collection of blocks and each block is persisted to the block table.

    NOTE: we are assuming agent.update_state has already been called.
    """

    for block_dict in agent.memory.to_dict()["memory"].values():
        # TODO: block creation should happen in one place to enforce these sort of constraints consistently.
        if block_dict.get("user_id", None) is None:
            block_dict["user_id"] = agent.agent_state.user_id
        block = Block(**block_dict)
        # FIXME: should we expect for block values to be None? If not, we need to figure out why that is
        # the case in some tests, if so we should relax the DB constraint.
        if block.value is None:
            block.value = ""
        ms.update_or_create_block(block)


def strip_name_field_from_user_message(user_message_text: str) -> Tuple[str, Optional[str]]:
    """If 'name' exists in the JSON string, remove it and return the cleaned text + name value"""
    try:
        user_message_json = dict(json_loads(user_message_text))
        # Special handling for AutoGen messages with 'name' field
        # Treat 'name' as a special field
        # If it exists in the input message, elevate it to the 'message' level
        name = user_message_json.pop("name", None)
        clean_message = json_dumps(user_message_json)
        return clean_message, name

    except Exception as e:
        print(f"{CLI_WARNING_PREFIX}handling of 'name' field failed with: {e}")
        raise e


def validate_json(user_message_text: str) -> str:
    """Make sure that the user input message is valid JSON"""
    try:
        user_message_json = dict(json_loads(user_message_text))
        user_message_json_val = json_dumps(user_message_json)
        return user_message_json_val
    except Exception as e:
        print(f"{CLI_WARNING_PREFIX}couldn't parse user input message as JSON: {e}")
        raise e
