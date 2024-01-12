---
title: koboldcpp
excerpt: Setting up MemGPT with koboldcpp
category: 6580da9a40bb410016b8b0c3
---

1. Download + install [koboldcpp](https://github.com/LostRuins/koboldcpp/) and the model you want to test with
2. In your terminal, run `./koboldcpp.py <MODEL> -contextsize <CONTEXT_LENGTH>`

For example, if we downloaded the model `dolphin-2.2.1-mistral-7b.Q6_K.gguf` and put it inside `~/models/TheBloke/`, we would run:

```sh
# using `-contextsize 8192` because Dolphin Mistral 7B has a context length of 8000 (and koboldcpp wants specific intervals, 8192 is the closest)
# the default port is 5001
./koboldcpp.py ~/models/TheBloke/dolphin-2.2.1-mistral-7B-GGUF/dolphin-2.2.1-mistral-7b.Q6_K.gguf --contextsize 8192
```

In your terminal where you're running MemGPT, run `memgpt configure` to set the default backend for MemGPT to point at koboldcpp:

```text
# if you are running koboldcpp locally, the default IP address + port will be http://localhost:5001
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): koboldcpp
? Enter default endpoint: http://localhost:5001
...
```

If you have an existing agent that you want to move to the koboldcpp backend, add extra flags to `memgpt run`:

```sh
memgpt run --agent your_agent --model-endpoint-type koboldcpp --model-endpoint http://localhost:5001
```
