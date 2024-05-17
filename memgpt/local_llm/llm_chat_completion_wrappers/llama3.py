import json

from memgpt.constants import JSON_ENSURE_ASCII, JSON_LOADS_STRICT
from memgpt.errors import LLMJSONParsingError
from memgpt.local_llm.json_parser import clean_json
from memgpt.local_llm.llm_chat_completion_wrappers.wrapper_base import (
    LLMChatCompletionWrapper,
)

PREFIX_HINT = """# Reminders:
# Important information about yourself and the user is stored in (limited) core memory
# You can modify core memory with core_memory_replace
# You can add to core memory with core_memory_append
# Less important information is stored in (unlimited) archival memory
# You can add to archival memory with archival_memory_insert
# You can search archival memory with archival_memory_search
# You will always see the statistics of archival memory, so you know if there is content inside it
# If you receive new important information about the user (or yourself), you immediately update your memory with core_memory_replace, core_memory_append, or archival_memory_insert"""

FIRST_PREFIX_HINT = """# Reminders:
# This is your first interaction with the user!
# Initial information about them is provided in the core memory user block
# Make sure to introduce yourself to them
# Your inner thoughts should be private, interesting, and creative
# Do NOT use inner thoughts to communicate with the user
# Use send_message to communicate with the user"""
# Don't forget to use send_message, otherwise the user won't see your message"""


class LLaMA3InnerMonologueWrapper(LLMChatCompletionWrapper):
    """ChatML-style prompt formatter, tested for use with https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct"""

    supports_first_message = True

    def __init__(
        self,
        json_indent=2,
        # simplify_json_content=True,
        simplify_json_content=False,
        clean_function_args=True,
        include_assistant_prefix=True,
        assistant_prefix_extra='\n{\n  "function":',
        assistant_prefix_extra_first_message='\n{\n  "function": "send_message",',
        allow_custom_roles=True,  # allow roles outside user/assistant
        use_system_role_in_user=False,  # use the system role on user messages that don't use "type: user_message"
        # allow_function_role=True,  # use function role for function replies?
        allow_function_role=False,  # use function role for function replies?
        no_function_role_role="assistant",  # if no function role, which role to use?
        no_function_role_prefix="FUNCTION RETURN:\n",  # if no function role, what prefix to use?
        # add a guiding hint
        assistant_prefix_hint=False,
    ):
        self.simplify_json_content = simplify_json_content
        self.clean_func_args = clean_function_args
        self.include_assistant_prefix = include_assistant_prefix
        self.assistant_prefix_extra = assistant_prefix_extra
        self.assistant_prefix_extra_first_message = assistant_prefix_extra_first_message
        self.assistant_prefix_hint = assistant_prefix_hint

        # role-based
        self.allow_custom_roles = allow_custom_roles
        self.use_system_role_in_user = use_system_role_in_user
        self.allow_function_role = allow_function_role
        # extras for when the function role is disallowed
        self.no_function_role_role = no_function_role_role
        self.no_function_role_prefix = no_function_role_prefix

        # how to set json in prompt
        self.json_indent = json_indent

    def _compile_function_description(self, schema, add_inner_thoughts=True) -> str:
        """Go from a JSON schema to a string description for a prompt"""
        # airorobos style
        func_str = ""
        func_str += f"{schema['name']}:"
        func_str += f"\n  description: {schema['description']}"
        func_str += "\n  params:"
        if add_inner_thoughts:
            from memgpt.local_llm.constants import (
                INNER_THOUGHTS_KWARG,
                INNER_THOUGHTS_KWARG_DESCRIPTION,
            )

            func_str += f"\n    {INNER_THOUGHTS_KWARG}: {INNER_THOUGHTS_KWARG_DESCRIPTION}"
        for param_k, param_v in schema["parameters"]["properties"].items():
            # TODO we're ignoring type
            func_str += f"\n    {param_k}: {param_v['description']}"
        # TODO we're ignoring schema['parameters']['required']
        return func_str

    def _compile_function_block(self, functions) -> str:
        """functions dict -> string describing functions choices"""
        prompt = ""

        # prompt += f"\nPlease select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format."
        prompt += "Please select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format."
        prompt += "\nAvailable functions:"
        for function_dict in functions:
            prompt += f"\n{self._compile_function_description(function_dict)}"

        return prompt

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def _compile_system_message(self, system_message, functions, function_documentation=None) -> str:
        """system prompt + memory + functions -> string"""
        prompt = ""
        prompt += system_message
        prompt += "\n"
        if function_documentation is not None:
            prompt += "Please select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format."
            prompt += "\nAvailable functions:\n"
            prompt += function_documentation
        else:
            prompt += self._compile_function_block(functions)
        return prompt

    def _compile_function_call(self, function_call, inner_thoughts=None):
        """Go from ChatCompletion to Airoboros style function trace (in prompt)

        ChatCompletion data (inside message['function_call']):
            "function_call": {
                "name": ...
                "arguments": {
                    "arg1": val1,
                    ...
                }

        Airoboros output:
            {
                "function": "send_message",
                "params": {
                "message": "Hello there! I am Sam, an AI developed by Liminal Corp. How can I assist you today?"
                }
            }
        """
        airo_func_call = {
            "function": function_call["name"],
            "params": {
                "inner_thoughts": inner_thoughts,
                **json.loads(function_call["arguments"], strict=JSON_LOADS_STRICT),
            },
        }
        return json.dumps(airo_func_call, indent=self.json_indent, ensure_ascii=JSON_ENSURE_ASCII)

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def _compile_assistant_message(self, message) -> str:
        """assistant message -> string"""
        prompt = ""

        # need to add the function call if there was one
        inner_thoughts = message["content"]
        if "function_call" in message and message["function_call"]:
            prompt += f"\n{self._compile_function_call(message['function_call'], inner_thoughts=inner_thoughts)}"
        elif "tool_calls" in message and message["tool_calls"]:
            for tool_call in message["tool_calls"]:
                prompt += f"\n{self._compile_function_call(tool_call['function'], inner_thoughts=inner_thoughts)}"
        else:
            # TODO should we format this into JSON somehow?
            prompt += inner_thoughts

        return prompt

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def _compile_user_message(self, message) -> str:
        """user message (should be JSON) -> string"""
        prompt = ""
        if self.simplify_json_content:
            # Make user messages not JSON but plaintext instead
            try:
                user_msg_json = json.loads(message["content"], strict=JSON_LOADS_STRICT)
                user_msg_str = user_msg_json["message"]
            except:
                user_msg_str = message["content"]
        else:
            # Otherwise just dump the full json
            try:
                user_msg_json = json.loads(message["content"], strict=JSON_LOADS_STRICT)
                user_msg_str = json.dumps(
                    user_msg_json,
                    indent=self.json_indent,
                    ensure_ascii=JSON_ENSURE_ASCII,
                )
            except:
                user_msg_str = message["content"]

        prompt += user_msg_str
        return prompt

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def _compile_function_response(self, message) -> str:
        """function response message (should be JSON) -> string"""
        # TODO we should clean up send_message returns to avoid cluttering the prompt
        prompt = ""
        try:
            # indent the function replies
            function_return_dict = json.loads(message["content"], strict=JSON_LOADS_STRICT)
            function_return_str = json.dumps(
                function_return_dict,
                indent=self.json_indent,
                ensure_ascii=JSON_ENSURE_ASCII,
            )
        except:
            function_return_str = message["content"]

        prompt += function_return_str
        return prompt

    def chat_completion_to_prompt(self, messages, functions, first_message=False, function_documentation=None):
        """chatml-style prompt formatting, with implied support for multi-role"""
        prompt = "<|begin_of_text|>"

        # System insturctions go first
        assert messages[0]["role"] == "system"
        system_block = self._compile_system_message(
            system_message=messages[0]["content"],
            functions=functions,
            function_documentation=function_documentation,
        )
        prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{system_block.strip()}<|eot_id|>"

        # Last are the user/assistant messages
        for message in messages[1:]:
            assert message["role"] in ["user", "assistant", "function", "tool"], message

            if message["role"] == "user":
                # Support for AutoGen naming of agents
                role_str = message["name"].strip().lower() if (self.allow_custom_roles and "name" in message) else message["role"]
                msg_str = self._compile_user_message(message)

                if self.use_system_role_in_user:
                    try:
                        msg_json = json.loads(message["content"], strict=JSON_LOADS_STRICT)
                        if msg_json["type"] != "user_message":
                            role_str = "system"
                    except:
                        pass
                prompt += f"\n<|start_header_id|>{role_str}<|end_header_id|>\n\n{msg_str.strip()}<|eot_id|>"

            elif message["role"] == "assistant":
                # Support for AutoGen naming of agents
                role_str = message["name"].strip().lower() if (self.allow_custom_roles and "name" in message) else message["role"]
                msg_str = self._compile_assistant_message(message)

                prompt += f"\n<|start_header_id|>{role_str}<|end_header_id|>\n\n{msg_str.strip()}<|eot_id|>"

            elif message["role"] in ["tool", "function"]:
                if self.allow_function_role:
                    role_str = message["role"]
                    msg_str = self._compile_function_response(message)
                    prompt += f"\n<|start_header_id|>{role_str}<|end_header_id|>\n\n{msg_str.strip()}<|eot_id|>"
                else:
                    # TODO figure out what to do with functions if we disallow function role
                    role_str = self.no_function_role_role
                    msg_str = self._compile_function_response(message)
                    func_resp_prefix = self.no_function_role_prefix
                    # NOTE whatever the special prefix is, it should also be a stop token
                    prompt += f"\n<|start_header_id|>{role_str}\n{func_resp_prefix}{msg_str.strip()}<|eot_id|>"

            else:
                raise ValueError(message)

        if self.include_assistant_prefix:
            prompt += "\n<|start_header_id|>assistant\n\n"
            if self.assistant_prefix_hint:
                prompt += f"\n{FIRST_PREFIX_HINT if first_message else PREFIX_HINT}"
            if self.supports_first_message and first_message:
                if self.assistant_prefix_extra_first_message:
                    prompt += self.assistant_prefix_extra_first_message
            else:
                if self.assistant_prefix_extra:
                    # assistant_prefix_extra='\n{\n  "function":',
                    prompt += self.assistant_prefix_extra

        return prompt

    def _clean_function_args(self, function_name, function_args):
        """Some basic MemGPT-specific cleaning of function args"""
        cleaned_function_name = function_name
        cleaned_function_args = function_args.copy() if function_args is not None else {}

        if function_name == "send_message":
            # strip request_heartbeat
            cleaned_function_args.pop("request_heartbeat", None)

        inner_thoughts = None
        if "inner_thoughts" in function_args:
            inner_thoughts = cleaned_function_args.pop("inner_thoughts")

        # TODO more cleaning to fix errors LLM makes
        return inner_thoughts, cleaned_function_name, cleaned_function_args

    def output_to_chat_completion_response(self, raw_llm_output, first_message=False):
        """Turn raw LLM output into a ChatCompletion style response with:
        "message" = {
            "role": "assistant",
            "content": ...,
            "function_call": {
                "name": ...
                "arguments": {
                    "arg1": val1,
                    ...
                }
            }
        }
        """
        # if self.include_opening_brance_in_prefix and raw_llm_output[0] != "{":
        # raw_llm_output = "{" + raw_llm_output
        assistant_prefix = self.assistant_prefix_extra_first_message if first_message else self.assistant_prefix_extra
        if assistant_prefix and raw_llm_output[: len(assistant_prefix)] != assistant_prefix:
            # print(f"adding prefix back to llm, raw_llm_output=\n{raw_llm_output}")
            raw_llm_output = assistant_prefix + raw_llm_output
            # print(f"->\n{raw_llm_output}")

        try:
            # cover llama.cpp server for now #TODO remove this when fixed
            raw_llm_output = raw_llm_output.rstrip()
            if raw_llm_output.endswith("<|eot_id|>"):
                raw_llm_output = raw_llm_output[: -len("<|eot_id|>")]
            function_json_output = clean_json(raw_llm_output)
        except Exception as e:
            raise Exception(f"Failed to decode JSON from LLM output:\n{raw_llm_output} - error\n{str(e)}")
        try:
            # NOTE: weird bug can happen where 'function' gets nested if the prefix in the prompt isn't abided by
            if isinstance(function_json_output["function"], dict):
                function_json_output = function_json_output["function"]
            # regular unpacking
            function_name = function_json_output["function"]
            function_parameters = function_json_output["params"]
        except KeyError as e:
            raise LLMJSONParsingError(
                f"Received valid JSON from LLM, but JSON was missing fields: {str(e)}. JSON result was:\n{function_json_output}"
            )

        if self.clean_func_args:
            (
                inner_thoughts,
                function_name,
                function_parameters,
            ) = self._clean_function_args(function_name, function_parameters)

        message = {
            "role": "assistant",
            "content": inner_thoughts,
            "function_call": {
                "name": function_name,
                "arguments": json.dumps(function_parameters, ensure_ascii=JSON_ENSURE_ASCII),
            },
        }
        return message
