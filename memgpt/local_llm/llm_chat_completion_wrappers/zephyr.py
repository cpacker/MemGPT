import json

from .wrapper_base import LLMChatCompletionWrapper
from ..json_parser import clean_json
from ...errors import LLMJSONParsingError


class ZephyrMistralWrapper(LLMChatCompletionWrapper):
    """
    Wrapper for Zephyr Alpha and Beta, Mistral 7B:
    https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
    https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    Note: this wrapper formats a prompt that only generates JSON, no inner thoughts
    """

    def __init__(
        self,
        simplify_json_content=True,
        clean_function_args=True,
        include_assistant_prefix=True,
        include_opening_brace_in_prefix=True,
        include_section_separators=False,
    ):
        self.simplify_json_content = simplify_json_content
        self.clean_func_args = clean_function_args
        self.include_assistant_prefix = include_assistant_prefix
        self.include_opening_brance_in_prefix = include_opening_brace_in_prefix
        self.include_section_separators = include_section_separators

    def chat_completion_to_prompt(self, messages, functions):
        """
        Zephyr prompt format:
            <|system|>
            </s>
            <|user|>
            {prompt}</s>
            <|assistant|>
        (source: https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF#prompt-template-zephyr)
        """

        prompt = ""

        IM_START_TOKEN = "<s>"
        IM_END_TOKEN = "</s>"

        # System instructions go first
        assert messages[0]["role"] == "system"
        prompt += f"<|system|>"
        prompt += f"\n{messages[0]['content']}"

        # Next is the functions preamble
        def create_function_description(schema):
            # airorobos style
            func_str = ""
            func_str += f"{schema['name']}:"
            func_str += f"\n  description: {schema['description']}"
            func_str += f"\n  params:"
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

        # Put functions INSIDE system message (TODO experiment with this)
        prompt += IM_END_TOKEN

        def create_function_call(function_call):
            airo_func_call = {
                "function": function_call["name"],
                "params": json.loads(function_call["arguments"]),
            }
            return json.dumps(airo_func_call, indent=2)

        for message in messages[1:]:
            assert message["role"] in ["user", "assistant", "function"], message

            if message["role"] == "user":
                if self.simplify_json_content:
                    try:
                        content_json = json.loads(message["content"])
                        content_simple = content_json["message"]
                        prompt += f"\n<|user|>\n{content_simple}{IM_END_TOKEN}"
                        # prompt += f"\nUSER: {content_simple}"
                    except:
                        prompt += f"\n<|user|>\n{message['content']}{IM_END_TOKEN}"
                        # prompt += f"\nUSER: {message['content']}"
            elif message["role"] == "assistant":
                prompt += f"\n<|assistant|>"
                if message["content"] is not None:
                    prompt += f"\n{message['content']}"
                # prompt += f"\nASSISTANT: {message['content']}"
                # need to add the function call if there was one
                if "function_call" in message and message["function_call"]:
                    prompt += f"\n{create_function_call(message['function_call'])}"
                prompt += f"{IM_END_TOKEN}"
            elif message["role"] == "function":
                # TODO find a good way to add this
                # prompt += f"\nASSISTANT: (function return) {message['content']}"
                prompt += f"\n<|assistant|>"
                prompt += f"\nFUNCTION RETURN: {message['content']}"
                # prompt += f"\nFUNCTION RETURN: {message['content']}"
                continue
            else:
                raise ValueError(message)

        # Add a sep for the response
        # if self.include_section_separators:
        # prompt += "\n### RESPONSE"

        if self.include_assistant_prefix:
            # prompt += f"\nASSISTANT:"
            prompt += f"\n<|assistant|>"
            if self.include_opening_brance_in_prefix:
                prompt += "\n{"

        return prompt

    def clean_function_args(self, function_name, function_args):
        """Some basic MemGPT-specific cleaning of function args"""
        cleaned_function_name = function_name
        cleaned_function_args = function_args.copy()

        if function_name == "send_message":
            # strip request_heartbeat
            cleaned_function_args.pop("request_heartbeat", None)

        # TODO more cleaning to fix errors LLM makes
        return cleaned_function_name, cleaned_function_args

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
        if self.include_opening_brance_in_prefix and raw_llm_output[0] != "{":
            raw_llm_output = "{" + raw_llm_output

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
            function_name, function_parameters = self.clean_function_args(function_name, function_parameters)

        message = {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": function_name,
                "arguments": json.dumps(function_parameters),
            },
        }
        return message


class ZephyrMistralInnerMonologueWrapper(ZephyrMistralWrapper):
    """Still expect only JSON outputs from model, but add inner monologue as a field"""

    """
    Wrapper for Zephyr Alpha and Beta, Mistral 7B:
    https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha
    https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
    Note: this wrapper formats a prompt with inner thoughts included
    """

    def __init__(
        self,
        simplify_json_content=True,
        clean_function_args=True,
        include_assistant_prefix=True,
        include_opening_brace_in_prefix=True,
        include_section_separators=True,
    ):
        self.simplify_json_content = simplify_json_content
        self.clean_func_args = clean_function_args
        self.include_assistant_prefix = include_assistant_prefix
        self.include_opening_brance_in_prefix = include_opening_brace_in_prefix
        self.include_section_separators = include_section_separators

    def chat_completion_to_prompt(self, messages, functions):
        prompt = ""

        IM_START_TOKEN = "<s>"
        IM_END_TOKEN = "</s>"

        # System insturctions go first
        assert messages[0]["role"] == "system"
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
            airo_func_call = {
                "function": function_call["name"],
                "params": {
                    "inner_thoughts": inner_thoughts,
                    **json.loads(function_call["arguments"]),
                },
            }
            return json.dumps(airo_func_call, indent=2)

        # Add a sep for the conversation
        if self.include_section_separators:
            prompt += "\n<|user|>"

        # Last are the user/assistant messages
        for message in messages[1:]:
            assert message["role"] in ["user", "assistant", "function"], message

            if message["role"] == "user":
                if self.simplify_json_content:
                    try:
                        content_json = json.loads(message["content"])
                        content_simple = content_json["message"]
                        prompt += f"\n<|user|>\n{content_simple}{IM_END_TOKEN}"
                    except:
                        prompt += f"\n<|user|>\n{message['content']}{IM_END_TOKEN}"
            elif message["role"] == "assistant":
                prompt += f"\n<|assistant|>"
                # need to add the function call if there was one
                inner_thoughts = message["content"]
                if "function_call" in message and message["function_call"]:
                    prompt += f"\n{create_function_call(message['function_call'], inner_thoughts=inner_thoughts)}"
            elif message["role"] == "function":
                # TODO find a good way to add this
                # prompt += f"\nASSISTANT: (function return) {message['content']}"
                prompt += f"\nFUNCTION RETURN: {message['content']}"
                continue
            else:
                raise ValueError(message)

        # Add a sep for the response
        # if self.include_section_separators:
        #    prompt += "\n### RESPONSE"

        if self.include_assistant_prefix:
            prompt += f"\n<|assistant|>"
            if self.include_opening_brance_in_prefix:
                prompt += "\n{"

        return prompt

    def clean_function_args(self, function_name, function_args):
        """Some basic MemGPT-specific cleaning of function args"""
        cleaned_function_name = function_name
        cleaned_function_args = function_args.copy()

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
        if self.include_opening_brance_in_prefix and raw_llm_output[0] != "{":
            raw_llm_output = "{" + raw_llm_output

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
