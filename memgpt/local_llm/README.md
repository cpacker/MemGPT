⁉️ Need help configuring local LLMs with MemGPT? Ask for help on [our Discord](https://discord.gg/9GEQrxmVyE) or [post on the GitHub discussion](https://github.com/cpacker/MemGPT/discussions/67).

👀 If you have a hosted ChatCompletion-compatible endpoint that works with function calling, you can simply set `OPENAI_API_BASE` (`export OPENAI_API_BASE=...`) to the IP+port of your endpoint. **As of 10/22/2023, most ChatCompletion endpoints do *NOT* support function calls, so if you want to play with MemGPT and open models, you probably need to follow the instructions below.**

🙋 Our examples assume that you're using [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui) to put your LLMs behind a web server. If you need help setting this up, check the instructions [here](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui). More LLM web server support to come soon (tell us what you use and we'll add it)!

---

# How to connect MemGPT to non-OpenAI LLMs

**If you have an LLM that is function-call finetuned**:
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

## Example with [Airoboros](https://huggingface.co/jondurbin/airoboros-l2-70b-2.1) (llama2 finetune)

To help you get started, we've implemented an example wrapper class for a popular llama2 model **finetuned on function calling** (Airoboros). We want MemGPT to run well on open models as much as you do, so we'll be actively updating this page with more examples. Additionally, we welcome contributions from the community! If you find an open LLM that works well with MemGPT, please open a PR with a model wrapper and we'll merge it ASAP.

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
See full file [here](llm_chat_completion_wrappers/airoboros.py). WebUI exposes a lot of parameters that can dramatically change LLM outputs, to change these you can modify the [WebUI settings file](/memgpt/local_llm/webui/settings.py).

### Running the example

```sh
# running airoboros behind a textgen webui server
export OPENAI_API_BASE = <pointing at webui server>
export BACKEND_TYPE = webui

# using --no_verify because this airoboros example does not output inner monologue, just functions
# airoboros is able to properly call `send_message`
$ python3 main.py --no_verify

Running... [exit by typing '/exit']
💭 Bootup sequence complete. Persona activated. Testing messaging functionality.

💭 None
🤖 Welcome! My name is Sam. How can I assist you today?
Enter your message: My name is Brad, not Chad...

💭 None
⚡🧠 [function] updating memory with core_memory_replace:
         First name: Chad
        → First name: Brad
```

---

## Status of ChatCompletion w/ function calling and open LLMs

MemGPT uses function calling to do memory management. With [OpenAI's ChatCompletion API](https://platform.openai.com/docs/api-reference/chat/), you can pass in a function schema in the `functions` keyword arg, and the API response will include a `function_call` field that includes the function name and the function arguments (generated JSON). How this works under the hood is your `functions` keyword is combined with the `messages` and `system` to form one big string input to the transformer, and the output of the transformer is parsed to extract the JSON function call.

In the future, more open LLMs and LLM servers (that can host OpenAI-compatable ChatCompletion endpoints) may start including parsing code to do this automatically as standard practice. However, in the meantime, when you see a model that says it supports “function calling”, like Airoboros, it doesn't mean that you can just load Airoboros into a ChatCompletion-compatable endpoint like WebUI, and then use the same OpenAI API call and it'll just work.

1. When a model page says it supports function calling, they probably mean that the model was finetuned on some function call data (not that you can just use ChatCompletion with functions out-of-the-box). Remember, LLMs are just string-in-string-out, so there are many ways to format the function call data. E.g. Airoboros formats the function schema in YAML style (see https://huggingface.co/jondurbin/airoboros-l2-70b-3.1.2#agentfunction-calling) and the output is in JSON style. To get this to work behind a ChatCompletion API, you still have to do the parsing from `functions` keyword arg (containing the schema) to the model's expected schema style in the prompt (YAML for Airoboros), and you have to run some code to extract the function call (JSON for Airoboros) and package it cleanly as a `function_call` field in the response.

2. Partly because of how complex it is to support function calling, most (all?) of the community projects that do OpenAI ChatCompletion endpoints for arbitrary open LLMs do not support function calling, because if they did, they would need to write model-specific parsing code for each one.

## What is this all this extra code for?

Because of the poor state of function calling support in existing ChatCompletion API serving code, we instead provide a light wrapper on top of ChatCompletion that adds parsers to handle function calling support. These parsers need to be specific to the model you're using (or at least specific to the way it was trained on function calling). We hope that our example code will help the community add additional compatability of MemGPT with more function-calling LLMs - we will also add more model support as we test more models and find those that work well enough to run MemGPT's function set.

To run the example of MemGPT with Airoboros, you'll need to host the model behind some LLM web server (for example [webui](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui)). Then, all you need to do is point MemGPT to this API endpoint by setting the environment variables `OPENAI_API_BASE` and `BACKEND_TYPE`. Now, instead of calling ChatCompletion on OpenAI's API, MemGPT will use it's own ChatCompletion wrapper that parses the system, messages, and function arguments into a format that Airoboros has been finetuned on, and once Airoboros generates a string output, MemGPT will parse the response to extract a potential function call (knowing what we know about Airoboros expected function call output).
