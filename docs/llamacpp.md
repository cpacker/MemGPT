---
title: llama.cpp
excerpt: Setting up MemGPT with llama.cpp
category: 6580da9a40bb410016b8b0c3
---

1. Download + install [llama.cpp](https://github.com/ggerganov/llama.cpp) and the model you want to test with
2. In your terminal, run `./server -m <MODEL> -c <CONTEXT_LENGTH>`

For example, if we downloaded the model `dolphin-2.2.1-mistral-7b.Q6_K.gguf` and put it inside `~/models/TheBloke/`, we would run:

```sh
# using `-c 8000` because Dolphin Mistral 7B has a context length of 8000
# the default port is 8080, you can change this with `--port`
./server -m ~/models/TheBloke/dolphin-2.2.1-mistral-7B-GGUF/dolphin-2.2.1-mistral-7b.Q6_K.gguf -c 8000
```

In your terminal where you're running MemGPT, run `memgpt configure` to set the default backend for MemGPT to point at llama.cpp:

```text
# if you are running llama.cpp locally, the default IP address + port will be http://localhost:8080
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): llamacpp
? Enter default endpoint: http://localhost:8080
...
```

If you have an existing agent that you want to move to the llama.cpp backend, add extra flags to `memgpt run`:

```sh
memgpt run --agent your_agent --model-endpoint-type llamacpp --model-endpoint http://localhost:8080
```
