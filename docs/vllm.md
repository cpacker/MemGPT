---
title: vLLM
excerpt: Setting up MemGPT with vLLM
category: 6580da9a40bb410016b8b0c3
---

1. Download + install [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html)
2. Launch a vLLM **OpenAI-compatible** API server using [the official vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)

For example, if we want to use the model `dolphin-2.2.1-mistral-7b` from [HuggingFace](https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b), we would run:

```sh
python -m vllm.entrypoints.openai.api_server \
--model ehartford/dolphin-2.2.1-mistral-7b
```

vLLM will automatically download the model (if it's not already downloaded) and store it in your [HuggingFace cache directory](https://huggingface.co/docs/datasets/cache).

In your terminal where you're running MemGPT, run `memgpt configure` to set the default backend for MemGPT to point at vLLM:

```text
# if you are running vLLM locally, the default IP address + port will be http://localhost:8000
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): vllm
? Enter default endpoint: http://localhost:8000
? Enter HuggingFace model tag (e.g. ehartford/dolphin-2.2.1-mistral-7b): ehartford/dolphin-2.2.1-mistral-7b
...
```

If you have an existing agent that you want to move to the vLLM backend, add extra flags to `memgpt run`:

```sh
memgpt run --agent your_agent --model-endpoint-type vllm --model-endpoint http://localhost:8000 --model ehartford/dolphin-2.2.1-mistral-7b
```
