import json

from .wrapper_base import LLMChatCompletionWrapper
from ..json_parser import clean_json
from ...errors import LLMJSONParsingError


class ChatMLInnerMonologueWrapper(LLMChatCompletionWrapper):
    """ChatML-style prompt formatter, tested for use with https://huggingface.co/ehartford/dolphin-2.5-mixtral-8x7b#training"""

    def __init__(
        self,
        simplify_json_content=True,
        clean_function_args=True,
        include_assistant_prefix=True,
        # include_opening_brace_in_prefix=True,
        # assistant_prefix_extra="\n{"
        # assistant_prefix_extra='\n{\n  "function": ',
        assistant_prefix_extra='\n{\n  "function":',
        include_section_separators=True,
    ):
        self.simplify_json_content = simplify_json_content
        self.clean_func_args = clean_function_args
        self.include_assistant_prefix = include_assistant_prefix
        # self.include_opening_brance_in_prefix = include_opening_brace_in_prefix
        self.assistant_prefix_extra = assistant_prefix_extra
        self.include_section_separators = include_section_separators

    def chat_completion_to_prompt(self, messages, functions):
        """Example for airoboros: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#prompt-format"""
        prompt = ""

        # System insturctions go first
        assert messages[0]["role"] == "system"

        prompt += "<|im_start|>system\n"
        prompt += messages[0]["content"]

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
        prompt += f"\nPlease select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format."
        prompt += f"\nAvailable functions:"
        for function_dict in functions:
            prompt += f"\n{create_function_description(function_dict)}"

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
            return json.dumps(airo_func_call, indent=2)

        # Add a sep for the conversation
        # if self.include_section_separators:
        #     prompt += "\n### INPUT"
        prompt += "<|im_end|>"

        # Last are the user/assistant messages
        for message in messages[1:]:
            assert message["role"] in ["user", "assistant", "function"], message

            if message["role"] == "user":
                # Support for AutoGen naming of agents
                if "name" in message:
                    user_prefix = message["name"].strip()
                    # user_prefix = f"USER ({user_prefix})"
                    user_prefix = f"<|im_start|>{user_prefix.lower()}"
                else:
                    # user_prefix = "USER"
                    user_prefix = "<|im_start|>user"
                if self.simplify_json_content:
                    try:
                        content_json = json.loads(message["content"])
                        content_simple = content_json["message"]
                        # prompt += f"\n{user_prefix}: {content_simple}"
                        prompt += f"\n{user_prefix}\n{content_simple}"
                    except:
                        # prompt += f"\n{user_prefix}: {message['content']}"
                        prompt += f"\n{user_prefix}\n{message['content']}"
                prompt += "<|im_end|>"
            elif message["role"] == "assistant":
                # Support for AutoGen naming of agents
                if "name" in message:
                    assistant_prefix = message["name"].strip()
                    # assistant_prefix = f"ASSISTANT ({assistant_prefix})"
                    assistant_prefix = f"<|im_start|>{assistant_prefix.lower()}"
                else:
                    # assistant_prefix = "ASSISTANT"
                    assistant_prefix = "<|im_start|>assistant"
                # prompt += f"\n{assistant_prefix}:"
                prompt += f"\n{assistant_prefix}"
                # need to add the function call if there was one
                inner_thoughts = message["content"]
                if "function_call" in message and message["function_call"]:
                    prompt += f"\n{create_function_call(message['function_call'], inner_thoughts=inner_thoughts)}"
                prompt += "<|im_end|>"
            elif message["role"] == "function":
                # TODO find a good way to add this
                # prompt += f"\nASSISTANT: (function return) {message['content']}"
                # prompt += f"\nFUNCTION RETURN: {message['content']}"
                prompt += f"\n<|im_start|>function\n{message['content']}"
                prompt += "<|im_end|>"
                continue
            else:
                raise ValueError(message)

        # # Add a sep for the response
        # if self.include_section_separators:
        #     prompt += "\n### RESPONSE"

        if self.include_assistant_prefix:
            # prompt += f"\nASSISTANT:"
            prompt += f"\n<|im_start|>assistant"
            if self.assistant_prefix_extra:
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

    def output_to_chat_completion_response(self, raw_llm_output):
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
        if self.assistant_prefix_extra and raw_llm_output[: len(self.assistant_prefix_extra)] != self.assistant_prefix_extra:
            # print(f"adding prefix back to llm, raw_llm_output=\n{raw_llm_output}")
            raw_llm_output = self.assistant_prefix_extra + raw_llm_output
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
