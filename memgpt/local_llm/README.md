## How to connect MemGPT to non-OpenAI LLMs

**If you have a hosted ChatCompletion-compatible endpoint that works with function calling**:
  - simply set `OPENAI_API_BASE` to the IP+port of your endpoint:

```sh
export OPENAI_API_BASE=...
```

For this to work, the endpoint **MUST** support function calls.

**As of 10/22/2023, most ChatCompletion endpoints do *NOT* support function calls, so if you want to play with MemGPT and open models, you probably need to follow the instructions below.**

## Integrating a function-call finetuned LLM with MemGPT

**If you have a hosted local model that is function-call finetuned**:
  - Implement a wrapper class for that model
    - The wrapper class needs to implement two functions:
      - One to go from ChatCompletion messages/functions schema to a prompt string
      - And one to go from raw LLM outputs to a ChatCompletion response
  - Put that model behind a server (e.g. using WebUI) and set `OPENAI_API_BASE`

```python
class LLMChatCompletionWrapper(ABC):

    @abstractmethod
    def chat_completion_to_prompt(self, messages, functions):
        """Go from ChatCompletion to a single prompt string"""
        pass

    @abstractmethod
    def output_to_chat_completion_response(self, raw_llm_output):
        """Turn the LLM output string into a ChatCompletion response"""
        pass
```

## Example with Airoboros LLM

To help you get started, we've implemented an example wrapper class for a popular llama2 model finetuned on function calling (airoboros). We want MemGPT to run well on open models as much as you do, so we'll be actively updating this page with more examples. Additionally, we welcome contributions from the community! If you find an open LLM that works well with MemGPT, please open a PR with a model wrapper and we'll merge it ASAP.

```python
class Airoboros21Wrapper(LLMChatCompletionWrapper):
    """Wrapper for Airoboros 70b v2.1: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1"""

    def chat_completion_to_prompt(self, messages, functions):
        """
        Examples for how airoboros expects its prompt inputs: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#prompt-format
        Examples for how airoboros expects to see function schemas: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#agentfunction-calling
        """

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
```
See full file [here](llm_chat_completion_wrappers/airoboros.py).

---

## Status of ChatCompletion w/ function calling and open LLMs

MemGPT uses function calling to do memory management. With OpenAI's ChatCompletion API, you can pass in a function schema in the `functions` keyword arg, and the API response will include a `function_call` field that includes the function name and the function arguments (generated JSON). How this works under the hood is your `functions` keyword is combined with the `messages` and `system` to form one big string input to the transformer, and the output of the transformer is parsed to extract the JSON function call.

In the future, more open LLMs and LLM servers (that can host OpenAI-compatable ChatCompletion endpoints) may start including parsing code to do this automatically as standard practice. However, in the meantime, when you see a model that says it supports “function calling”, like Airoboros, it doesn't mean that you can just load Airoboros into a ChatCompletion-compatable endpoint like FastChat, and then use the same OpenAI API call and it'll just work.

1. When an open LLM says it supports function calling, they probably mean that the model was finetuned on some function call data. Remember, transformers are just string-in-string-out, so there are many ways to format this function call data. Airoboros formats the function schema in YAML style (see https://huggingface.co/jondurbin/airoboros-l2-70b-3.1.2#agentfunction-calling) and the output is in JSON style. To get this to work behind a ChatCompletion API, you still have to do the parsing from ‘functions’ keyword arg (containing the schema) to the model's expected schema style in the prompt (YAML for Airoboros), and you have to run some code to extract the function call (JSON for Airoboros) and package it cleanly as a ‘function_call’ field in the response.

2. Partly because of how complex it is to support function calling, most (all?) of the community projects that do OpenAI ChatCompletion endpoints for arbitrary open LLMs do not support function calling, because if they did, they would need to write model-specific parsing code for each one.

## How can you run MemGPT with open LLMs that support function calling?

Because of the poor state of function calling support in existing ChatCompletion API serving code, we instead provide a light wrapper on top of ChatCompletion that uses a parser specific to Airoboros. We hope that this example code will help the community add additional compatability of MemGPT with more function-calling LLMs - we will also add more model support as we test more models and find those that work well enough to run MemGPT's function set.

To run the example of MemGPT with Airoboros, you'll need to host the model with some open LLM hosting code, for example Oobagooba (see here). Then, all you need to do is point MemGPT to this API endpoint. Now, instead of calling ChatCompletion on OpenAI's API, MemGPT will use it's own ChatCompletion wrapper that parses the system, messages, and function arguments into a format that Airoboros has been finetuned on, and once Airoboros generates a string output, MemGPT will parse the response to extract a potential function call (knowing what we know about Airoboros expected function call output).
