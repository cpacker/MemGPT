import json
from typing import List

from .wrapper_base import LLMChatCompletionWrapper


# A configurable model agnostic wrapper.
class ConfigurableWrapper(LLMChatCompletionWrapper):
    def __init__(self, pre_prompt: str,
                 post_prompt: str,
                 sys_prompt_start: str,
                 sys_prompt_end: str,
                 user_prompt_start: str,
                 user_prompt_end: str,
                 assistant_prompt_start: str,
                 assistant_prompt_end: str,
                 function_prompt_start: str,
                 function_prompt_end: str,
                 include_sys_prompt_in_first_user_message: bool,
                 default_stop_sequences: List[str],

                 strip_prompt: bool = True,
                 json_indent: int = 2):
        """
        Initializes a new MessagesFormatter object.

        Args:
            pre_prompt (str): The pre-prompt content.
            sys_prompt_start (str): The system prompt start.
            sys_prompt_end (str): The system prompt end.
            user_prompt_start (str): The user prompt start.
            user_prompt_end (str): The user prompt end.
            assistant_prompt_start (str): The assistant prompt start.
            assistant_prompt_end (str): The assistant prompt end.
            include_sys_prompt_in_first_user_message (bool): Indicates whether to include the system prompt
                                                             in the first user message.
            default_stop_sequences (List[str]): List of default stop sequences.
            function_prompt_start (str): The function prompt start.
            function_prompt_end (str): The function prompt end.
        """
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.sys_prompt_start = sys_prompt_start
        self.sys_prompt_end = sys_prompt_end
        self.user_prompt_start = user_prompt_start
        self.user_prompt_end = user_prompt_end
        self.assistant_prompt_start = assistant_prompt_start
        self.assistant_prompt_end = assistant_prompt_end

        self.include_sys_prompt_in_first_user_message = include_sys_prompt_in_first_user_message
        self.default_stop_sequences = default_stop_sequences
        self.function_prompt_start = function_prompt_start
        self.function_prompt_end = function_prompt_end
        self.strip_prompt = strip_prompt
        self.json_indent = json_indent

    def chat_completion_to_prompt(self, messages, functions, function_documentation=None):
        formatted_messages = self.pre_prompt
        last_role = "assistant"
        no_user_prompt_start = False
        for message in messages:
            if message["role"] == "system":
                formatted_messages += self.sys_prompt_start + message["content"] + self.sys_prompt_end
                last_role = "system"
                if self.include_sys_prompt_in_first_user_message:
                    formatted_messages = self.user_prompt_start + formatted_messages
                    no_user_prompt_start = True
            elif message["role"] == "user":
                if no_user_prompt_start:
                    no_user_prompt_start = False
                    formatted_messages += message["content"] + self.user_prompt_end
                else:
                    formatted_messages += self.user_prompt_start + message["content"] + self.user_prompt_end
                last_role = "user"
            elif message["role"] == "assistant":
                formatted_messages += self.assistant_prompt_start + message["content"] + self.assistant_prompt_end
                last_role = "assistant"
            elif message["role"] == "function":
                message["content"] = json.dumps(message["content"], indent=self.json_indent)
                formatted_messages += self.function_prompt_start + message["content"] + self.function_prompt_end
                last_role = "function"
        if last_role == "system" or last_role == "user" or last_role == "function":
            if self.strip_prompt:
                return formatted_messages + self.assistant_prompt_start.strip(), "assistant"
            else:
                return formatted_messages + self.assistant_prompt_start, "assistant"
        if self.strip_prompt:
            return formatted_messages + self.user_prompt_start.strip(), "user"
        else:
            return formatted_messages + self.user_prompt_start, "user"

    def output_to_chat_completion_response(self, raw_llm_output):
        pass
