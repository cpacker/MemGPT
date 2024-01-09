import json

from .wrapper_base import LLMChatCompletionWrapper
from ..json_parser import clean_json
from ...errors import LLMJSONParsingError


class Airoboros21Wrapper(LLMChatCompletionWrapper):
    """Wrapper for Airoboros 70b v2.1: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1

    Note: this wrapper formats a prompt that only generates JSON, no inner thoughts
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
        """Example for airoboros: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#prompt-format

        A chat.
        USER: {prompt}
        ASSISTANT:

        Functions support: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#agentfunction-calling

            As an AI assistant, please select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format.

            Input: I want to know how many times 'Python' is mentioned in my text file.

            Available functions:
            file_analytics:
              description: This tool performs various operations on a text file.
              params:
                action: The operation we want to perform on the data, such as "count_occurrences", "find_line", etc.
                filters:
                  keyword: The word or phrase we want to search for.

        OpenAI functions schema style:

            {
                "name": "send_message",
                "description": "Sends a message to the human user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # https://json-schema.org/understanding-json-schema/reference/array.html
                        "message": {
                            "type": "string",
                            "description": "Message contents. All unicode (including emojis) are supported.",
                        },
                    },
                    "required": ["message"],
                }
            },
        """
        prompt = ""

        # System insturctions go first
        assert messages[0]["role"] == "system"
        prompt += messages[0]["content"]

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

        def create_function_call(function_call):
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
                "params": json.loads(function_call["arguments"]),
            }
            return json.dumps(airo_func_call, indent=2, ensure_ascii=False)

        # Add a sep for the conversation
        if self.include_section_separators:
            prompt += "\n### INPUT"

        # Last are the user/assistant messages
        for message in messages[1:]:
            assert message["role"] in ["user", "assistant", "function"], message

            if message["role"] == "user":
                if self.simplify_json_content:
                    try:
                        content_json = json.loads(message["content"])
                        content_simple = content_json["message"]
                        prompt += f"\nUSER: {content_simple}"
                    except:
                        prompt += f"\nUSER: {message['content']}"
            elif message["role"] == "assistant":
                prompt += f"\nASSISTANT: {message['content']}"
                # need to add the function call if there was one
                if "function_call" in message and message["function_call"]:
                    prompt += f"\n{create_function_call(message['function_call'])}"
            elif message["role"] == "function":
                # TODO find a good way to add this
                # prompt += f"\nASSISTANT: (function return) {message['content']}"
                prompt += f"\nFUNCTION RETURN: {message['content']}"
                continue
            else:
                raise ValueError(message)

        # Add a sep for the response
        if self.include_section_separators:
            prompt += "\n### RESPONSE"

        if self.include_assistant_prefix:
            prompt += f"\nASSISTANT:"
            if self.include_opening_brance_in_prefix:
                prompt += "\n{"

        print(prompt)
        return prompt

    def clean_function_args(self, function_name, function_args):
        """Some basic MemGPT-specific cleaning of function args"""
        cleaned_function_name = function_name
        cleaned_function_args = function_args.copy() if function_args is not None else {}

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
                "arguments": json.dumps(function_parameters, ensure_ascii=False),
            },
        }
        return message


class Airoboros21InnerMonologueWrapper(Airoboros21Wrapper):
    """Still expect only JSON outputs from model, but add inner monologue as a field"""

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
        """Example for airoboros: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#prompt-format

        A chat.
        USER: {prompt}
        ASSISTANT:

        Functions support: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#agentfunction-calling

            As an AI assistant, please select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format.

            Input: I want to know how many times 'Python' is mentioned in my text file.

            Available functions:
            file_analytics:
              description: This tool performs various operations on a text file.
              params:
                action: The operation we want to perform on the data, such as "count_occurrences", "find_line", etc.
                filters:
                  keyword: The word or phrase we want to search for.

        OpenAI functions schema style:

            {
                "name": "send_message",
                "description": "Sends a message to the human user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # https://json-schema.org/understanding-json-schema/reference/array.html
                        "message": {
                            "type": "string",
                            "description": "Message contents. All unicode (including emojis) are supported.",
                        },
                    },
                    "required": ["message"],
                }
            },
        """
        prompt = ""

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
            return json.dumps(airo_func_call, indent=2, ensure_ascii=False)

        # Add a sep for the conversation
        if self.include_section_separators:
            prompt += "\n### INPUT"

        # Last are the user/assistant messages
        for message in messages[1:]:
            assert message["role"] in ["user", "assistant", "function"], message

            if message["role"] == "user":
                # Support for AutoGen naming of agents
                if "name" in message:
                    user_prefix = message["name"].strip()
                    user_prefix = f"USER ({user_prefix})"
                else:
                    user_prefix = "USER"
                if self.simplify_json_content:
                    try:
                        content_json = json.loads(message["content"])
                        content_simple = content_json["message"]
                        prompt += f"\n{user_prefix}: {content_simple}"
                    except:
                        prompt += f"\n{user_prefix}: {message['content']}"
            elif message["role"] == "assistant":
                # Support for AutoGen naming of agents
                if "name" in message:
                    assistant_prefix = message["name"].strip()
                    assistant_prefix = f"ASSISTANT ({assistant_prefix})"
                else:
                    assistant_prefix = "ASSISTANT"
                prompt += f"\n{assistant_prefix}:"
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
        if self.include_section_separators:
            prompt += "\n### RESPONSE"

        if self.include_assistant_prefix:
            prompt += f"\nASSISTANT:"
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
            # NOTE: weird bug can happen where 'function' gets nested if the prefix in the prompt isn't abided by
            if isinstance(function_json_output["function"], dict):
                function_json_output = function_json_output["function"]
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
            ) = self.clean_function_args(function_name, function_parameters)

        message = {
            "role": "assistant",
            "content": inner_thoughts,
            "function_call": {
                "name": function_name,
                "arguments": json.dumps(function_parameters, ensure_ascii=False),
            },
        }
        return message
