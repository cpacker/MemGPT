import json
from .wrapper_base import LLMChatCompletionWrapper


class SimpleSummaryWrapper(LLMChatCompletionWrapper):
    """A super basic wrapper that's meant to be used for summary generation only"""

    def __init__(
        self,
        simplify_json_content=True,
        include_assistant_prefix=True,
        # include_assistant_prefix=False,  # False here, because we launch directly into summary
        include_section_separators=True,
    ):
        self.simplify_json_content = simplify_json_content
        self.include_assistant_prefix = include_assistant_prefix
        self.include_section_separators = include_section_separators

    def chat_completion_to_prompt(self, messages, functions):
        """Example for airoboros: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#prompt-format

        Instructions on how to summarize
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
        assert functions is None
        prompt = ""

        # System insturctions go first
        assert messages[0]["role"] == "system"
        prompt += messages[0]["content"]

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
            return json.dumps(airo_func_call, indent=2)

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
                if message["function_call"]:
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
            prompt += "\n### RESPONSE (your summary of the above conversation in plain English (no JSON!), do NOT exceed the word limit)"

        if self.include_assistant_prefix:
            # prompt += f"\nASSISTANT:"
            prompt += f"\nSUMMARY:"

        # print(prompt)
        return prompt

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
        raw_llm_output = raw_llm_output.strip()
        message = {
            "role": "assistant",
            "content": raw_llm_output,
            # "function_call": {
            # "name": function_name,
            # "arguments": json.dumps(function_parameters),
            # },
        }
        return message
