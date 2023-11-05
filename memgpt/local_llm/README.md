⁉️ Need help configuring local LLMs with MemGPT? Ask for help on [our Discord](https://discord.gg/9GEQrxmVyE) or [post on the GitHub discussion](https://github.com/cpacker/MemGPT/discussions/67).

If you have a hosted ChatCompletion-compatible endpoint that works with function calling, you can simply set `OPENAI_API_BASE` (`export OPENAI_API_BASE=...`) to the IP+port of your endpoint. **As of 10/22/2023, most ChatCompletion endpoints do *NOT* support function calls, so if you want to play with MemGPT and open models, you probably need to follow the instructions below.**

---

# ⚡ Quick overview

1. Put your own LLM behind a web server API (e.g. [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui))
2. Set `OPENAI_API_BASE=YOUR_API_IP_ADDRESS` and `BACKEND_TYPE=webui`
3. Run MemGPT with `python3 main.py --no_verify`, it should now use your LLM instead of OpenAI GPT
4. If things aren't working, read the full instructions below

When using open LLMs with MemGPT, **the main failure case will be your LLM outputting a string that cannot be understood by MemGPT**. MemGPT uses function calling to manage memory (eg `edit_core_memory(...)` and interact with the user (`send_message(...)`), so your LLM needs generate outputs that can be parsed into MemGPT function calls.

---

# How to connect MemGPT to non-OpenAI LLMs

<details>
 <summary><h2>🖥️ Serving your LLM from a web server (WebUI example)</strong></h2></summary>

⁉️ Do **NOT** enable any extensions in web UI, including the [openai extension](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)! Just run web UI as-is, unless you are running [MemGPT+Autogen](https://github.com/cpacker/MemGPT/tree/main/memgpt/autogen) with non-MemGPT agents.

To get MemGPT to work with a local LLM, you need to have the LLM running on a server that takes API requests.

For the purposes of this example, we're going to serve (host) the LLMs using [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui), but if you want to use something else you can! This also assumes your running web UI locally - if you're running on e.g. Runpod, you'll want to follow Runpod specific instructions (for example use [TheBloke's one-click UI and API](https://github.com/TheBlokeAI/dockerLLM/blob/main/README_Runpod_LocalLLMsUIandAPI.md))

1. Install oobabooga web UI using the instructions [here](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui)
2. Once installed, launch the web server with `python server.py`
3. Navigate to the web app (if local, this is probably [`http://127.0.0.1:7860`](http://localhost:7860)), select the model you want to use, adjust your GPU and CPU memory settings, and click "load"
4. If the model was loaded successfully, you should be able to access it via the API (if local, this is probably on port `5000`)
5. Assuming steps 1-4 went correctly, the LLM is now properly hosted on a port you can point MemGPT to!

In your terminal where you're running MemGPT, run:

```sh
# if you are running web UI locally, the default port will be 5000
export OPENAI_API_BASE=http://127.0.0.1:5000
export BACKEND_TYPE=webui
```

WebUI exposes a lot of parameters that can dramatically change LLM outputs, to change these you can modify the [WebUI settings file](/memgpt/local_llm/webui/settings.py).

⁉️ If you have problems getting WebUI setup, please use the [official web UI repo for support](https://github.com/oobabooga/text-generation-webui)! There will be more answered questions about web UI there vs here on the MemGPT repo.

</details>

<details>
 <summary><h2>🖥️ Serving your LLM from a web server (LM Studio example)</strong></h2></summary>

![image](https://github.com/cpacker/MemGPT/assets/5475622/abc8ce2d-4130-4c51-8169-83e682db625d)

1. Download [LM Studio](https://lmstudio.ai/) and the model you want to test with
2. Go to the "local inference server" tab, load the model and configure your settings (make sure to set the context length to something reasonable like 8k!)
3. Click "Start server"
4. Copy the IP address + port that your server is running on (in the example screenshot, the address is `http://localhost:1234`)

In your terminal where you're running MemGPT, run:

```sh
# if you used a different port in LM Studio, change 1234 to the actual port
export OPENAI_API_BASE=http://localhost:1234
export BACKEND_TYPE=lmstudio
```

</details>

<details>
 <summary><h2>🦙 Running MemGPT with your own LLM</strong></h2></summary>

Once you have an LLM web server set up, all you need to do to connect it to MemGPT is set two environment variables:

- `OPENAI_API_BASE`
  - set this to the IP address of your LLM API - for example, if you're using web UI on a local machine, this will look like `http://127.0.0.1:5000`
- `BACKEND_TYPE`
  - set this to `webui` or `lmstudio`
  - this controls how MemGPT packages the HTTP request to the webserver, see [this code](https://github.com/cpacker/MemGPT/blob/main/memgpt/local_llm/webui/api.py)
  - currently this is set up to work with web UI, but it might work with other backends / web servers too!
  - if you'd like to use a different web server and you need a different style of HTTP request, let us know on the discussion page (https://github.com/cpacker/MemGPT/discussions/67) and we'll try to add it ASAP

You can change the prompt format and output parser used with the `--model` flag. For example:

```sh
# this will cause MemGPT to use the airoboros-l2-70b-2.1 parsers, regardless of what model you're hosting on your web server
# you can mix and match parsers + models!
$ python3 main.py --model airoboros-l2-70b-2.1
```

### Example with airoboros 70b

```sh
# assuming we're running a model (eg airoboros) behind a textgen webui server
export OPENAI_API_BASE=127.0.0.1:5000  # change this to your actual API address
export BACKEND_TYPE=webui  # if you don't set this, MemGPT will throw an error

# using --no_verify can be helpful if the LLM you're using doesn't output inner monologue properly
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

</details>

<details>
 <summary><h2>🙋 Adding support for new LLMs + improving performance</strong></h2></summary>

⁉️ When using open LLMs with MemGPT, **the main failure case will be your LLM outputting a string that cannot be understood by MemGPT**. MemGPT uses function calling to manage memory (eg `edit_core_memory(...)`) and interact with the user (`send_message(...)`), so your LLM needs generate outputs that can be parsed into MemGPT function calls.

### What is a "wrapper"?

To support function calling with open LLMs for MemGPT, we utilize "wrapper" code that:

1. turns `system` (the MemGPT instructions), `messages` (the MemGPT conversation window), and `functions` (the MemGPT function set) parameters from ChatCompletion into a single unified prompt string for your LLM
2. turns the output string generated by your LLM back into a MemGPT function call

Different LLMs are trained using different prompt formats (eg `#USER:` vs `<im_start>user` vs ...), and LLMs that are trained on function calling are often trained using different function call formats, so if you're getting poor performance, try experimenting with different prompt formats! We recommend starting with the prompt format (and function calling format) recommended in the HuggingFace model card, and experimenting from there.

We currently only support a few prompt formats in this repo ([located here](https://github.com/cpacker/MemGPT/tree/main/memgpt/local_llm/llm_chat_completion_wrappers))! If you write a new parser, please open a PR and we'll merge it in.

<details>
 <summary><h3>Adding a new wrapper (change the prompt format + function parser)</strong></h3></summary>

To make a new wrapper (for example, because you want to try a different prompt format), you just need to subclass `LLMChatCompletionWrapper`. Your new wrapper class needs to implement two functions:

- One to go from ChatCompletion messages/functions schema to a prompt string
- And one to go from raw LLM outputs to a ChatCompletion response

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

You can follow our example wrappers ([located here](https://github.com/cpacker/MemGPT/tree/main/memgpt/local_llm/llm_chat_completion_wrappers)).

</details>

<details>
 <summary><h3>Example wrapper for Airoboros</strong></h3></summary>

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

See full file [here](llm_chat_completion_wrappers/airoboros.py).

</details>

</details>

---

## FAQ

<details>
 <summary><h3>Status of ChatCompletion w/ function calling and open LLMs</strong></h3></summary>

MemGPT uses function calling to do memory management. With [OpenAI's ChatCompletion API](https://platform.openai.com/docs/api-reference/chat/), you can pass in a function schema in the `functions` keyword arg, and the API response will include a `function_call` field that includes the function name and the function arguments (generated JSON). How this works under the hood is your `functions` keyword is combined with the `messages` and `system` to form one big string input to the transformer, and the output of the transformer is parsed to extract the JSON function call.

In the future, more open LLMs and LLM servers (that can host OpenAI-compatable ChatCompletion endpoints) may start including parsing code to do this automatically as standard practice. However, in the meantime, when you see a model that says it supports “function calling”, like Airoboros, it doesn't mean that you can just load Airoboros into a ChatCompletion-compatable endpoint like WebUI, and then use the same OpenAI API call and it'll just work.

1. When a model page says it supports function calling, they probably mean that the model was finetuned on some function call data (not that you can just use ChatCompletion with functions out-of-the-box). Remember, LLMs are just string-in-string-out, so there are many ways to format the function call data. E.g. Airoboros formats the function schema in YAML style (see https://huggingface.co/jondurbin/airoboros-l2-70b-3.1.2#agentfunction-calling) and the output is in JSON style. To get this to work behind a ChatCompletion API, you still have to do the parsing from `functions` keyword arg (containing the schema) to the model's expected schema style in the prompt (YAML for Airoboros), and you have to run some code to extract the function call (JSON for Airoboros) and package it cleanly as a `function_call` field in the response.

2. Partly because of how complex it is to support function calling, most (all?) of the community projects that do OpenAI ChatCompletion endpoints for arbitrary open LLMs do not support function calling, because if they did, they would need to write model-specific parsing code for each one.

</details>

<details>
 <summary><h3>What is this all this extra code for?</strong></h3></summary>

Because of the poor state of function calling support in existing ChatCompletion API serving code, we instead provide a light wrapper on top of ChatCompletion that adds parsers to handle function calling support. These parsers need to be specific to the model you're using (or at least specific to the way it was trained on function calling). We hope that our example code will help the community add additional compatability of MemGPT with more function-calling LLMs - we will also add more model support as we test more models and find those that work well enough to run MemGPT's function set.

To run the example of MemGPT with Airoboros, you'll need to host the model behind some LLM web server (for example [webui](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui)). Then, all you need to do is point MemGPT to this API endpoint by setting the environment variables `OPENAI_API_BASE` and `BACKEND_TYPE`. Now, instead of calling ChatCompletion on OpenAI's API, MemGPT will use it's own ChatCompletion wrapper that parses the system, messages, and function arguments into a format that Airoboros has been finetuned on, and once Airoboros generates a string output, MemGPT will parse the response to extract a potential function call (knowing what we know about Airoboros expected function call output).

</details>

<details open>
 <summary><h3>Need more help?</h3></summary>

 Ask for help on [our Discord](https://discord.gg/9GEQrxmVyE) or [post on the GitHub discussion](https://github.com/cpacker/MemGPT/discussions/67).
</details>
