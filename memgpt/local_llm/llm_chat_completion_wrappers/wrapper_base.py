from abc import ABC, abstractmethod


class LLMChatCompletionWrapper(ABC):
    @abstractmethod
    def chat_completion_to_prompt(self, messages, functions, function_documentation=None):
        """Go from ChatCompletion to a single prompt string"""

    @abstractmethod
    def output_to_chat_completion_response(self, raw_llm_output):
        """Turn the LLM output string into a ChatCompletion response"""
