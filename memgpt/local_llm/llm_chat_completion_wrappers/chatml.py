import json

from .wrapper_base import LLMChatCompletionWrapper
from ..json_parser import clean_json
from ...errors import LLMJSONParsingError


class ChatMLInnerMonologueWrapper(LLMChatCompletionWrapper):
    """ChatML-style prompt formatter, tested for use with https://huggingface.co/ehartford/dolphin-2.5-mixtral-8x7b#training"""

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
        # allow_function_role=True,  # use function role for function replies?
        allow_function_role=False,  # use function role for function replies?
        no_function_role_role="assistant",  # if no function role, which role to use?
        no_function_role_prefix="FUNCTION RETURN:\n",  # if no function role, what prefix to use?
    ):
        self.simplify_json_content = simplify_json_content
        self.clean_func_args = clean_function_args
        self.include_assistant_prefix = include_assistant_prefix
        self.assistant_prefix_extra = assistant_prefix_extra
        self.assistant_prefix_extra_first_message = assistant_prefix_extra_first_message

        # role-based
        self.allow_custom_roles = allow_custom_roles
        self.allow_function_role = allow_function_role
        # extras for when the function role is disallowed
        self.no_function_role_role = no_function_role_role
        self.no_function_role_prefix = no_function_role_prefix

        # how to set json in prompt
        self.json_indent = json_indent

    def compile_function_block(self, functions) -> str:
        """functions dict -> string describing functions choices"""
        prompt = ""

        # Next is the functions preamble
        def create_function_description(schema, add_inner_thoughts=True):
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

        # prompt += f"\nPlease select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format."
        prompt += f"Please select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format."
        prompt += f"\nAvailable functions:"
        for function_dict in functions:
            prompt += f"\n{create_function_description(function_dict)}"

        return prompt

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def compile_system_message(self, system_message, functions) -> str:
        """system prompt + memory + functions -> string"""
        prompt = ""
        prompt += system_message
        prompt += "\n"
        prompt += self.compile_function_block(functions)
        return prompt

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def compile_assistant_message(self, message) -> str:
        """assistant message -> string"""
        prompt = ""

        def create_function_call(function_call, inner_thoughts=None):
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
                    **json.loads(function_call["arguments"]),
                },
            }
            return json.dumps(airo_func_call, indent=self.json_indent)

        # need to add the function call if there was one
        inner_thoughts = message["content"]
        if "function_call" in message and message["function_call"]:
            prompt += f"\n{create_function_call(message['function_call'], inner_thoughts=inner_thoughts)}"
        else:
            # TODO should we format this into JSON somehow?
            prompt += inner_thoughts

        return prompt

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def compile_user_message(self, message) -> str:
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
                user_msg_str = json.dumps(user_msg_json, indent=self.json_indent)
            except:
                user_msg_str = message["content"]

        prompt += user_msg_str
        return prompt

    # NOTE: BOS/EOS chatml tokens are NOT inserted here
    def compile_function_response(self, message) -> str:
        """function response message (should be JSON) -> string"""
        # TODO we should clean up send_message returns to avoid cluttering the prompt
        prompt = ""
        try:
            # indent the function replies
            function_return_dict = json.loads(message["content"])
            function_return_str = json.dumps(function_return_dict, indent=self.json_indent)
        except:
            function_return_str = message["content"]

        prompt += function_return_str
        return prompt

    def chat_completion_to_prompt(self, messages, functions, first_message=False):
        """chatml-style prompt formatting, with implied support for multi-role"""
        prompt = ""

        # System insturctions go first
        assert messages[0]["role"] == "system"
        system_block = self.compile_system_message(system_message=messages[0]["content"], functions=functions)
        prompt += f"<|im_start|>system\n{system_block.strip()}<|im_end|>"

        # Last are the user/assistant messages
        for message in messages[1:]:
            assert message["role"] in ["user", "assistant", "function"], message

            if message["role"] == "user":
                # Support for AutoGen naming of agents
                role_str = message["name"].strip().lower() if (self.allow_custom_roles and "name" in message) else message["role"]
                msg_str = self.compile_user_message(message)

                prompt += f"\n<|im_start|>{role_str}\n{msg_str.strip()}<|im_end|>"

            elif message["role"] == "assistant":
                # Support for AutoGen naming of agents
                role_str = message["name"].strip().lower() if (self.allow_custom_roles and "name" in message) else message["role"]
                msg_str = self.compile_assistant_message(message)

                prompt += f"\n<|im_start|>{role_str}\n{msg_str.strip()}<|im_end|>"

            elif message["role"] == "function":
                if self.allow_function_role:
                    role_str = message["role"]
                    msg_str = self.compile_function_response(message)
                    prompt += f"\n<|im_start|>{role_str}\n{msg_str.strip()}<|im_end|>"
                else:
                    # TODO figure out what to do with functions if we disallow function role
                    role_str = self.no_function_role_role
                    msg_str = self.compile_function_response(message)
                    func_resp_prefix = self.no_function_role_prefix
                    # NOTE whatever the special prefix is, it should also be a stop token
                    prompt += f"\n<|im_start|>{role_str}\n{func_resp_prefix}{msg_str.strip()}<|im_end|>"

            else:
                raise ValueError(message)

        if self.include_assistant_prefix:
            prompt += f"\n<|im_start|>assistant"
            if first_message:
                if self.assistant_prefix_extra_first_message:
                    prompt += self.assistant_prefix_extra_first_message
            else:
                if self.assistant_prefix_extra:
                    # assistant_prefix_extra='\n{\n  "function":',
                    prompt += self.assistant_prefix_extra

        return prompt

    def clean_function_args(self, function_name, function_args):
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
            function_json_output = clean_json(raw_llm_output)
        except Exception as e:
            raise Exception(f"Failed to decode JSON from LLM output:\n{raw_llm_output} - error\n{str(e)}")
        try:
            function_name = function_json_output["function"]
            function_parameters = function_json_output["params"]
        except KeyError as e:
            raise LLMJSONParsingError(f"Received valid JSON from LLM, but JSON was missing fields: {str(e)}")

        if self.clean_func_args:
            (
                inner_thoughts,
                function_name,
                function_parameters,
            ) = self.clean_function_args(function_name, function_parameters)

        message = {
            "role": "assistant",
            "content": inner_thoughts,
            "function_call": {
                "name": function_name,
                "arguments": json.dumps(function_parameters),
            },
        }
        return message


class ChatMLOuterInnerMonologueWrapper(ChatMLInnerMonologueWrapper):
    """Moves the inner monologue outside the main function to allow the LLM to omit function calls

    NOTE: warning - this makes it easier for the agent to forget to call functions,
          so it is advised to use the function-forcing wrapper unless the LLM is very good
    """
