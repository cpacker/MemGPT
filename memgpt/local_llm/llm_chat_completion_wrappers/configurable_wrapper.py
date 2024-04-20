import json

import yaml

from ...constants import JSON_ENSURE_ASCII, JSON_LOADS_STRICT
from ...errors import LLMJSONParsingError
from ..json_parser import clean_json
from .wrapper_base import LLMChatCompletionWrapper


# A configurable model agnostic wrapper.
class ConfigurableJSONWrapper(LLMChatCompletionWrapper):
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
        assistant_prefix_extra="",
        assistant_prefix_extra_first_message="",
        allow_custom_roles: bool = False,  # allow roles outside user/assistant
        custom_post_role: str = "",  # For chatml this would be '\n'
        custom_roles_prompt_start: str = "",  # For chatml this would be '<|im_start|>'
        custom_roles_prompt_end: str = "",  # For chatml this would be '<|im_end|>'
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
            post_prompt (str): The post-prompt content
            sys_prompt_start (str): The system messages prompt start. For chatml, this would be '<|im_start|>system\n'
            sys_prompt_end (str): The system messages prompt end. For chatml, this would be '<|im_end|>'
            user_prompt_start (str): The user messages prompt start. For chatml, this would be '<|im_start|>user\n'
            user_prompt_end (str): The user messages prompt end. For chatml, this would be '<|im_end|>\n'
            assistant_prompt_start (str): The assistant messages prompt start. For chatml, this would be '<|im_start|>user\n'
            assistant_prompt_end (str): The assistant messages prompt end. For chatml, this would be '<|im_end|>\n'
            tool_prompt_start (str): The tool messages prompt start. For chatml, this would be '<|im_start|>tool\n' if the model supports the tool role, otherwise it would be something like '<|im_start|>user\nFUNCTION RETURN:\n'
            tool_prompt_end (str): The tool messages prompt end. For chatml, this would be '<|im_end|>\n'
            assistant_prefix_extra (str): A prefix for every assistant message to steer the model to output JSON. Something like '\n{\n  "function":'
            assistant_prefix_extra_first_message (str): A prefix for the first assistant message to steer the model to output JSON and use a specific function. Something like '\n{\n  "function": "send_message",'
            allow_custom_roles (bool): If the wrapper allows custom roles, like names for autogen agents.
            custom_post_role (str): The part that comes after the custom role string.  For chatml, this would be '\n'
            custom_roles_prompt_start: (str): Custom role prompt start. For chatml, this would be '<|im_start|>'
            custom_roles_prompt_end: (str): Custom role prompt start. For chatml, this would be '<|im_end|>\n'
            include_sys_prompt_in_first_user_message (bool): Indicates whether to include the system prompt in the first user message. For Llama2 this would be True, for chatml, this would be False
            simplify_json_content (bool):
            strip_prompt (bool): If whitespaces at the end and beginning of the prompt get stripped.
            default_stop_sequences (List[str]): List of default stop sequences.

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
        self.tool_prompt_start = tool_prompt_start
        self.tool_prompt_end = tool_prompt_end
        self.assistant_prefix_extra = assistant_prefix_extra
        self.assistant_prefix_extra_first_message = assistant_prefix_extra_first_message
        self.allow_custom_roles = allow_custom_roles
        self.custom_post_role = custom_post_role
        self.custom_roles_prompt_start = custom_roles_prompt_start
        self.custom_roles_prompt_end = custom_roles_prompt_end
        self.include_sys_prompt_in_first_user_message = include_sys_prompt_in_first_user_message
        self.simplify_json_content = simplify_json_content
        self.default_stop_sequences = default_stop_sequences
        self.strip_prompt = strip_prompt
        self.json_indent = json_indent
        self.clean_func_args = clean_function_args
        self.supports_first_message = True

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
        prompt = system_message
        prompt += "\n"
        if function_documentation is not None:
            prompt += f"Please select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format."
            prompt += f"\nAvailable functions:"
            prompt += function_documentation
        else:
            prompt += self._compile_function_block(functions)
        return prompt

    def _compile_function_call(self, function_call, inner_thoughts=None):
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
            function_return_dict = json.loads(message["content"], strict=JSON_LOADS_STRICT)
            function_return_str = json.dumps(function_return_dict, indent=self.json_indent, ensure_ascii=JSON_ENSURE_ASCII)
        except:
            function_return_str = message["content"]

        prompt += function_return_str
        return prompt

    def chat_completion_to_prompt(self, messages, functions, first_message=False, function_documentation=None):
        formatted_messages = self.pre_prompt

        no_user_prompt_start = False

        for message in messages:
            if message["role"] == "system":
                msg = self._compile_system_message(message["content"], functions, function_documentation)
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
                if self.allow_custom_roles and "name" in message:
                    role_str = message["name"].strip().lower() if (self.allow_custom_roles and "name" in message) else message["role"]
                    if no_user_prompt_start:
                        no_user_prompt_start = False
                        formatted_messages += (
                            self.user_prompt_end
                            + self.custom_roles_prompt_start
                            + role_str
                            + self.custom_post_role
                            + msg
                            + self.custom_roles_prompt_end
                        )
                    else:
                        formatted_messages += (
                            self.custom_roles_prompt_start + role_str + self.custom_post_role + msg + self.custom_roles_prompt_end
                        )
                else:
                    if no_user_prompt_start:
                        no_user_prompt_start = False
                        formatted_messages += self.user_prompt_end + self.assistant_prompt_start + msg + self.assistant_prompt_end
                    else:
                        formatted_messages += self.assistant_prompt_start + msg + self.assistant_prompt_end
            elif message["role"] == "tool":
                msg = self._compile_function_response(message)
                formatted_messages += self.tool_prompt_start + msg + self.tool_prompt_end

        if self.strip_prompt:
            if first_message:
                prompt = formatted_messages + self.post_prompt + self.assistant_prefix_extra_first_message
            else:
                prompt = formatted_messages + self.post_prompt + self.assistant_prefix_extra
            return prompt.strip()
        else:
            if first_message:
                prompt = formatted_messages + self.post_prompt + self.assistant_prefix_extra_first_message
            else:
                prompt = formatted_messages + self.post_prompt + self.assistant_prefix_extra
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
        assistant_prefix = self.assistant_prefix_extra_first_message if first_message else self.assistant_prefix_extra
        if assistant_prefix and raw_llm_output[: len(assistant_prefix)] != assistant_prefix:
            raw_llm_output = assistant_prefix + raw_llm_output

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
            if "inner_thoughts" in function_json_output:
                inner_thoughts = function_json_output["inner_thoughts"]
            else:
                if "inner_thoughts" in function_json_output["params"]:
                    inner_thoughts = function_json_output["params"]["inner_thoughts"]
                else:
                    inner_thoughts = ""
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
        data = {
            "pre_prompt": self.pre_prompt,
            "post_prompt": self.post_prompt,
            "sys_prompt_start": self.sys_prompt_start,
            "sys_prompt_end": self.sys_prompt_end,
            "user_prompt_start": self.user_prompt_start,
            "user_prompt_end": self.user_prompt_end,
            "assistant_prompt_start": self.assistant_prompt_start,
            "assistant_prompt_end": self.assistant_prompt_end,
            "tool_prompt_start": self.tool_prompt_start,
            "tool_prompt_end": self.tool_prompt_end,
            "assistant_prefix_extra": self.assistant_prefix_extra,
            "assistant_prefix_extra_first_message": self.assistant_prefix_extra_first_message,
            "allow_custom_roles": self.allow_custom_roles,
            "custom_post_role": self.custom_post_role,
            "custom_roles_prompt_start": self.custom_roles_prompt_start,
            "custom_roles_prompt_end": self.custom_roles_prompt_end,
            "include_sys_prompt_in_first_user_message": self.include_sys_prompt_in_first_user_message,
            "simplify_json_content": self.simplify_json_content,
            "strip_prompt": self.strip_prompt,
            "json_indent": self.json_indent,
            "clean_function_args": self.clean_func_args,
            "default_stop_sequences": self.default_stop_sequences,
        }

        with open(file_path, "w") as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False)

    @staticmethod
    def load_from_yaml(file_path: str):
        """
        Load the configuration from a YAML file.

        Args:
            file_path (str): The path to the YAML file.
        """
        with open(file_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)

        wrapper = ConfigurableJSONWrapper()
        # Set the attributes from the loaded data
        wrapper.pre_prompt = data.get("pre_prompt", "")
        wrapper.post_prompt = data.get("post_prompt", "")
        wrapper.sys_prompt_start = data.get("sys_prompt_start", "")
        wrapper.sys_prompt_end = data.get("sys_prompt_end", "")
        wrapper.user_prompt_start = data.get("user_prompt_start", "")
        wrapper.user_prompt_end = data.get("user_prompt_end", "")
        wrapper.assistant_prompt_start = data.get("assistant_prompt_start", "")
        wrapper.assistant_prompt_end = data.get("assistant_prompt_end", "")
        wrapper.tool_prompt_start = data.get("tool_prompt_start", "")
        wrapper.tool_prompt_end = data.get("tool_prompt_end", "")
        wrapper.assistant_prefix_extra = data.get("assistant_prefix_extra", "")
        wrapper.assistant_prefix_extra_first_message = data.get("assistant_prefix_extra_first_message", "")
        wrapper.allow_custom_roles = data.get("allow_custom_roles", False)
        wrapper.custom_post_role = data.get("custom_post_role", "")
        wrapper.custom_roles_prompt_start = data.get("custom_roles_prompt_start", "")
        wrapper.custom_roles_prompt_end = data.get("custom_roles_prompt_end", "")
        wrapper.include_sys_prompt_in_first_user_message = data.get("include_sys_prompt_in_first_user_message", False)
        wrapper.simplify_json_content = data.get("simplify_json_content", False)
        wrapper.strip_prompt = data.get("strip_prompt", False)
        wrapper.json_indent = data.get("json_indent", 2)
        wrapper.clean_func_args = data.get("clean_function_args", False)
        wrapper.default_stop_sequences = data.get("default_stop_sequences", [])

        return wrapper
