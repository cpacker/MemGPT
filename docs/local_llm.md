## Using MemGPT with local LLMs

!!! warning "MemGPT + local LLM failure cases"

    When using open LLMs with MemGPT, **the main failure case will be your LLM outputting a string that cannot be understood by MemGPT**. MemGPT uses function calling to manage memory (eg `edit_core_memory(...)` and interact with the user (`send_message(...)`), so your LLM needs generate outputs that can be parsed into MemGPT function calls.

    Make sure to check the [local LLM troubleshooting page](../local_llm_faq) to see common issues before raising a new issue or posting on Discord.

### Installing dependencies
To install dependencies required for running local models, run:
```
pip install 'pymemgpt[local]'
```

### Quick overview

1. Put your own LLM behind a web server API (e.g. [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui))
2. Set `OPENAI_API_BASE=YOUR_API_IP_ADDRESS` and `BACKEND_TYPE=webui`

For example, if we are running web UI (which defaults to port 5000) on the same computer as MemGPT, we would do the following:
```sh
# set this to the backend we're using, eg 'webui', 'lmstudio', 'llamacpp', 'koboldcpp'
export BACKEND_TYPE=webui
# set this to the base address of llm web server
export OPENAI_API_BASE=http://127.0.0.1:5000
```

Now when we run MemGPT, it will use the LLM running on the local web server.

### Selecting a model wrapper

When you use local LLMs, `model` no longer specifies the LLM model that is run (you determine that yourself by loading a model in your backend interface). Instead, `model` refers to the _wrapper_ that is used to parse data sent to and from the LLM backend.

You can change the wrapper used with the `--model` flag. For example, the following :
```sh
memgpt run --model airoboros-l2-70b-2.1
```

The default wrapper is `airoboros-l2-70b-2.1-grammar` if you are using a backend that supports grammar-based sampling, and `airoboros-l2-70b-2.1` otherwise.

Note: the wrapper name does **not** have to match the model name. For example, the `dolphin-2.1-mistral-7b` model works better with the `airoboros-l2-70b-2.1` wrapper than the `dolphin-2.1-mistral-7b` wrapper. The model you load inside your LLM backend (e.g. LM Studio) determines what model is actually run, the `--model` flag just determines how the prompt is formatted before it is passed to the LLM backend.

### Grammars

Grammar-based sampling can help improve the performance of MemGPT when using local LLMs. Grammar-based sampling works by restricting the outputs of an LLM to a "grammar", for example, the MemGPT JSON function call grammar. Without grammar-based sampling, it is common to encounter JSON-related errors when using local LLMs with MemGPT.

To use grammar-based sampling, make sure you're using a backend that supports it: webui, llama.cpp, or koboldcpp, then you should specify one of the new wrappers that implements grammars, eg: `airoboros-l2-70b-2.1-grammar`.

### Supported backends

Currently, MemGPT supports the following backends:

* [oobabooga web UI](../webui) (Mac, Windows, Linux) (✔️ supports grammars)
* [LM Studio](../lmstudio) (Mac, Windows) (❌ does not support grammars)
* [koboldcpp](../koboldcpp) (Mac, Windows, Linux) (✔️ supports grammars)
* [llama.cpp](../llamacpp) (Mac, Windows, Linux) (✔️ supports grammars)

If you would like us to support a new backend, feel free to open an issue or pull request on [the MemGPT GitHub page](https://github.com/cpacker/MemGPT)!

### Which model should I use?

If you are experimenting with MemGPT and local LLMs for the first time, we recommend you try the Dolphin Mistral finetune (e.g. [ehartford/dolphin-2.2.1-mistral-7b](https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b) or a quantized variant such as [dolphin-2.2.1-mistral-7b.Q6_K.gguf](https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF)), and use the default `airoboros` wrapper.

Generating MemGPT-compatible outputs is a harder task for an LLM than regular text output. For this reason **we strongly advise users to NOT use models below Q5 quantization** - as the model gets worse, the number of errors you will encounter while using MemGPT will dramatically increase (MemGPT will not send messages properly, edit memory properly, etc.).

Check out [our local LLM GitHub discussion](https://github.com/cpacker/MemGPT/discussions/67) and [the MemGPT Discord server](https://discord.gg/9GEQrxmVyE) for more advice on model selection and help with local LLM troubleshooting.
