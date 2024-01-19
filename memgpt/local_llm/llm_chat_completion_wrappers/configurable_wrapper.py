import json
from typing import List

import yaml

from .wrapper_base import LLMChatCompletionWrapper
from ...constants import JSON_ENSURE_ASCII
from ..json_parser import clean_json
from ...constants import JSON_ENSURE_ASCII
from ...errors import LLMJSONParsingError


# A configurable model agnostic wrapper.
class ConfigurableWrapper(LLMChatCompletionWrapper):
    def __init__(
        self,
        pre_prompt: str = "",
        post_prompt: str = "",
        sys_prompt_start: str = "",
        sys_prompt_end: str = "",
        user_prompt_start: str = "",
        user_prompt_end: str = "",
        assistant_prompt_start: str = "",
        assistant_prompt_end: str = "",
        tool_prompt_start: str = "",
        tool_prompt_end: str = "",
        include_sys_prompt_in_first_user_message: bool = False,
        default_stop_sequences=None,
        simplify_json_content: bool = False,
        strip_prompt: bool = False,
        json_indent: int = 2,
        clean_function_args: bool = False,
    ):
        """
        Initializes a new MessagesFormatter object.

        Args:
            pre_prompt (str): The pre-prompt content.
            post_prompt(str): The post-prompt content
            sys_prompt_start (str): The system prompt start.
            sys_prompt_end (str): The system prompt end.
            user_prompt_start (str): The user prompt start.
            user_prompt_end (str): The user prompt end.
            assistant_prompt_start (str): The assistant prompt start.
            assistant_prompt_end (str): The assistant prompt end.
            include_sys_prompt_in_first_user_message (bool): Indicates whether to include the system prompt
                                                             in the first user message.
            simplify_json_content (bool):

            default_stop_sequences (List[str]): List of default stop sequences.
            tool_prompt_start (str): The tool prompt start.
            tool_prompt_end (str): The tool prompt end.
        """
        if default_stop_sequences is None:
            default_stop_sequences = []
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.sys_prompt_start = sys_prompt_start
        self.sys_prompt_end = sys_prompt_end
        self.user_prompt_start = user_prompt_start
        self.user_prompt_end = user_prompt_end
        self.assistant_prompt_start = assistant_prompt_start
        self.assistant_prompt_end = assistant_prompt_end
        self.include_sys_prompt_in_first_user_message = include_sys_prompt_in_first_user_message
        self.simplify_json_content = simplify_json_content
        self.default_stop_sequences = default_stop_sequences
        self.tool_prompt_start = tool_prompt_start
        self.tool_prompt_end = tool_prompt_end
        self.strip_prompt = strip_prompt
        self.json_indent = json_indent
        self.clean_func_args = clean_function_args

    def _compile_function_description(self, schema, add_inner_thoughts=True) -> str:
        """Go from a JSON schema to a string description for a prompt"""
        # airorobos style
        func_str = ""
        func_str += f"{schema['name']}:"
        func_str += f"\n  description: {schema['description']}"
        func_str += f"\n  params:"
        if add_inner_thoughts:
            func_str += f"\n    inner_thoughts: Deep inner monologue private to you only."
        for param_k, param_v in schema["parameters"]["properties"].items():
            # TODO we're ignoring type
            func_str += f"\n    {param_k}: {param_v['description']}"
        # TODO we're ignoring schema['parameters']['required']
        return func_str

    def _compile_function_block(self, functions) -> str:
        """functions dict -> string describing functions choices"""
        prompt = ""

        # prompt += f"\nPlease select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format."
        prompt += f"Please select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format."
        prompt += f"\nAvailable functions:"
        for function_dict in functions:
            prompt += f"\n{self._compile_function_description(function_dict)}"

        return prompt

    def _compile_system_message(self, system_message, functions, function_documentation=None) -> str:
        """system prompt + memory + functions -> string"""
        prompt = ""
        prompt += system_message
        prompt += "\n"
        if function_documentation is not None:
            prompt += function_documentation
        else:
            prompt += self._compile_function_block(functions)
        return prompt

    def _compile_function_call(self, function_call, inner_thoughts=None):
        airo_func_call = {
            "function": function_call["name"],
            "params": {
                "inner_thoughts": inner_thoughts,
                **json.loads(function_call["arguments"]),
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
                user_msg_json = json.loads(message["content"])
                user_msg_str = user_msg_json["message"]
            except:
                user_msg_str = message["content"]
        else:
            # Otherwise just dump the full json
            try:
                user_msg_json = json.loads(message["content"])
                user_msg_str = json.dumps(user_msg_json, indent=self.json_indent, ensure_ascii=JSON_ENSURE_ASCII)
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
            function_return_dict = json.loads(message["content"])
            function_return_str = json.dumps(function_return_dict, indent=self.json_indent, ensure_ascii=JSON_ENSURE_ASCII)
        except:
            function_return_str = message["content"]

        prompt += function_return_str
        return prompt

    def chat_completion_to_prompt(self, messages, functions, function_documentation=None):
        formatted_messages = self.pre_prompt

        no_user_prompt_start = False
        assert messages[0]["role"] == "system"
        for message in messages:
            assert message["role"] in ["user", "assistant", "tool"], message
            if message["role"] == "system":
                msg = self._compile_system_message(message, functions, function_documentation)
                formatted_messages += self.sys_prompt_start + msg + self.sys_prompt_end

                if self.include_sys_prompt_in_first_user_message:
                    formatted_messages = self.user_prompt_start + formatted_messages
                    no_user_prompt_start = True
            elif message["role"] == "user":
                msg = self._compile_user_message(message)
                if no_user_prompt_start:
                    no_user_prompt_start = False
                    formatted_messages += msg + self.user_prompt_end
                else:
                    formatted_messages += self.user_prompt_start + msg + self.user_prompt_end

            elif message["role"] == "assistant":
                msg = self._compile_assistant_message(message)
                formatted_messages += self.assistant_prompt_start + msg + self.assistant_prompt_end

            elif message["role"] == "tool":
                msg = self._compile_function_response(message)
                formatted_messages += self.tool_prompt_start + msg + self.tool_prompt_end
        if self.strip_prompt:
            prompt = formatted_messages + self.post_prompt
            return prompt.strip()
        else:
            return formatted_messages + self.post_prompt

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

    def output_to_chat_completion_response(self, raw_llm_output):
        try:
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

    def save_to_yaml(self, file_path: str):
        """
        Save the configuration to a YAML file.

        Args:
            file_path (str): The path to the YAML file.
        """
        config_data = {
            "pre_prompt": self.pre_prompt,
            "post_prompt": self.post_prompt,
            "sys_prompt_start": self.sys_prompt_start,
            "sys_prompt_end": self.sys_prompt_end,
            "user_prompt_start": self.user_prompt_start,
            "user_prompt_end": self.user_prompt_end,
            "assistant_prompt_start": self.assistant_prompt_start,
            "assistant_prompt_end": self.assistant_prompt_end,
            "include_sys_prompt_in_first_user_message": self.include_sys_prompt_in_first_user_message,
            "simplify_json_content": self.simplify_json_content,
            "default_stop_sequences": self.default_stop_sequences,
            "tool_prompt_start": self.tool_prompt_start,
            "tool_prompt_end": self.tool_prompt_end,
            "strip_prompt": self.strip_prompt,
            "json_indent": self.json_indent,
            "clean_function_args": self.clean_func_args,
        }

        with open(file_path, "w") as yaml_file:
            yaml.dump(config_data, yaml_file, default_flow_style=False)

    def load_from_yaml(self, file_path: str):
        """
        Load the configuration from a YAML file.

        Args:
            file_path (str): The path to the YAML file.
        """
        with open(file_path, "r") as yaml_file:
            config_data = yaml.safe_load(yaml_file)

        # Update the instance variables with the loaded data
        self.__dict__.update(config_data)
