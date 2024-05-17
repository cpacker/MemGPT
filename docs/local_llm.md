---
title: MemGPT + open models
excerpt: Set up MemGPT to run with open LLMs
category: 6580da9a40bb410016b8b0c3
---

> 📘 Need help?
>
> Visit our [Discord server](https://discord.gg/9GEQrxmVyE) and post in the #support channel. Make sure to check the [local LLM troubleshooting page](local_llm_faq) to see common issues before raising a new issue or posting on Discord.

> 📘 Using Windows?
>
> If you're using Windows and are trying to get MemGPT with local LLMs setup, we recommend using Anaconda Shell, or WSL (for more advanced users). See more Windows installation tips [here](local_llm_faq).

> ⚠️ MemGPT + open LLM failure cases
>
> When using open LLMs with MemGPT, **the main failure case will be your LLM outputting a string that cannot be understood by MemGPT**. MemGPT uses function calling to manage memory (eg `edit_core_memory(...)` and interact with the user (`send_message(...)`), so your LLM needs generate outputs that can be parsed into MemGPT function calls. See [the local LLM troubleshooting page](local_llm_faq) for more information.

### Installing dependencies

To install dependencies required for running local models, run:

```sh
pip install 'pymemgpt[local]'
```

If you installed from source (`git clone` then `pip install -e .`), do:

```sh
pip install -e '.[local]'
```

If you installed from source using Poetry, do:

```sh
poetry install -E local
```

### Quick overview

1. Put your own LLM behind a web server API (e.g. [llama.cpp server](https://github.com/ggerganov/llama.cpp/tree/master/examples/server#quick-start) or [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui))
2. Run `memgpt configure` and when prompted select your backend/endpoint type and endpoint address (a default will be provided but you may have to override it)

For example, if we are running web UI (which defaults to port 5000) on the same computer as MemGPT, running `memgpt configure` would look like this:

```text
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): webui
? Enter default endpoint: http://localhost:5000
? Select default model wrapper (optimal choice depends on specific llm, for llama3 we recommend llama3-grammar, for legacy llms it is airoboros-l2-70b-2.1): llama3-grammar
? Select your model's context window (for Mistral 7B models and Meta-Llama-3-8B-Instruct, this is probably 8k / 8192): 8192
? Select embedding provider: local
? Select default preset: memgpt_chat
? Select default persona: sam_pov
? Select default human: cs_phd
? Select storage backend for archival data: local
Saving config to /home/user/.memgpt/config
```

Now when we do `memgpt run`, it will use the LLM running on the local web server.

If you want to change the local LLM settings of an existing agent, you can pass flags to `memgpt run`:

```sh
# --model-wrapper will override the wrapper
# --model-endpoint will override the endpoint address
# --model-endpoint-type will override the backend type

# if we were previously using "agent_11" with web UI, and now want to use lmstudio, we can do:
memgpt run --agent agent_11 --model-endpoint http://localhost:1234 --model-endpoint-type lmstudio
```

### Selecting a model wrapper

When you use local LLMs, you can specify a **model wrapper** that changes how the LLM input text is formatted before it is passed to your LLM.

You can change the wrapper used with the `--model-wrapper` flag:

```sh
memgpt run --model-wrapper llama3-grammar
```

You can see the full selection of model wrappers by running `memgpt configure`:

```text
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): webui
? Enter default endpoint: http://localhost:5000
? Select default model wrapper (recommended: llama3-grammar for llama3 llms, airoboros-l2-70b-2.1 for legacy models): (Use arrow keys)
 » llama3
   llama3-grammar
   llama3-hints-grammar
   airoboros-l2-70b-2.1
   airoboros-l2-70b-2.1-grammar
   dolphin-2.1-mistral-7b
   dolphin-2.1-mistral-7b-grammar
   zephyr-7B
   zephyr-7B-grammar
```

Note: the wrapper name does **not** have to match the model name. For example, the `dolphin-2.1-mistral-7b` model works better with the `airoboros-l2-70b-2.1` wrapper than the `dolphin-2.1-mistral-7b` wrapper. The model you load inside your LLM backend (e.g. LM Studio) determines what model is actually run, the `--model-wrapper` flag just determines how the prompt is formatted before it is passed to the LLM backend.

### Grammars

Grammar-based sampling can help improve the performance of MemGPT when using local LLMs. Grammar-based sampling works by restricting the outputs of an LLM to a "grammar", for example, the MemGPT JSON function call grammar. Without grammar-based sampling, it is common to encounter JSON-related errors when using local LLMs with MemGPT.

To use grammar-based sampling, make sure you're using a backend that supports it: webui, llama.cpp, or koboldcpp, then you should specify one of the new wrappers that implements grammars, eg: `airoboros-l2-70b-2.1-grammar`.

Note that even though grammar-based sampling can reduce the mistakes your LLM makes, it can also make your model inference significantly slower.

### Supported backends

Currently, MemGPT supports the following backends:

* [oobabooga web UI](webui) (Mac, Windows, Linux) (✔️ supports grammars)
* [LM Studio](lmstudio) (Mac, Windows) (❌ does not support grammars)
* [koboldcpp](koboldcpp) (Mac, Windows, Linux) (✔️ supports grammars)
* [llama.cpp](llamacpp) (Mac, Windows, Linux) (✔️ supports grammars)
* [vllm](vllm) (Mac, Windows, Linux) (❌ does not support grammars)

If you would like us to support a new backend, feel free to open an issue or pull request on [the MemGPT GitHub page](https://github.com/cpacker/MemGPT)!

### Which model should I use?

> 📘 Recommended LLMs / models
>
> To see a list of recommended LLMs to use with MemGPT, visit our [Discord server](https://discord.gg/9GEQrxmVyE) and check the #model-chat channel.

Most recently, one of the best models to run locally is Meta's [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) or its quantized version such as [Meta-Llama-3-8B-Instruct-Q6_K.gguf](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF).

If you are experimenting with MemGPT and local LLMs for the first time, we recommend you try the Dolphin Mistral finetune (e.g. [ehartford/dolphin-2.2.1-mistral-7b](https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b) or a quantized variant such as [dolphin-2.2.1-mistral-7b.Q6_K.gguf](https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF)), and use the default `airoboros` wrapper.

Generating MemGPT-compatible outputs is a harder task for an LLM than regular text output. For this reason **we strongly advise users to NOT use models below Q5 quantization** - as the model gets worse, the number of errors you will encounter while using MemGPT will dramatically increase (MemGPT will not send messages properly, edit memory properly, etc.).

> 📘 Advanced LLMs / models
>
Enthusiasts with high-VRAM GPUS (3090,4090) or apple silicon macs with >32G VRAM might find [IQ2_XS quantization of Llama-3-70B](https://huggingface.co/MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF) interesting, as it is currently the highest-performing opensource/openweights model. You can run it in llama.cpp with setup such as this: `./server -m Meta-Llama-3-70B-Instruct.IQ2_XS.gguf --n-gpu-layers 99 --no-mmap --ctx-size 8192 -ctk q8_0 --chat-template llama3 --host 0.0.0.0 --port 8888`.
