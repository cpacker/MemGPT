---
title: LM Studio
excerpt: Setting up MemGPT with LM Studio
category: 6580da9a40bb410016b8b0c3
---

> 📘 Update your LM Studio
>
> The current `lmstudio` backend will only work if your LM Studio is version 0.2.9 or newer.
>
> If you are on a version of LM Studio older than 0.2.9 (<= 0.2.8), select `lmstudio-legacy` as your backend type.
>
> ⚠️ Important LM Studio settings
>
> **Context length**: Make sure that "context length" (`n_ctx`) is set (in "Model initialization" on the right hand side "Server Model Settings" panel) to the max context length of the model you're using (e.g. 8000 for Mistral 7B variants).
>
> **Automatic Prompt Formatting = OFF**: If you see "Automatic Prompt Formatting" inside LM Studio's "Server Options" panel (on the left side), turn it **OFF**. Leaving it **ON** will break MemGPT.
>
> **Context Overflow Policy = Stop at limit**: If you see "Context Overflow Policy" inside LM Studio's "Tools" panel on the right side (below "Server Model Settings"), set it to **Stop at limit**. The default setting "Keep the system prompt ... truncate middle" will break MemGPT.

<img width="911" alt="image" src="https://github.com/cpacker/MemGPT/assets/5475622/d499e82e-348c-4468-9ea6-fd15a13eb7fa">

1. Download [LM Studio](https://lmstudio.ai/) and the model you want to test with
2. Go to the "local inference server" tab, load the model and configure your settings (make sure to set the context length to something reasonable like 8k!)
3. Click "Start server"
4. Copy the IP address + port that your server is running on (in the example screenshot, the address is `http://localhost:1234`)

In your terminal where you're running MemGPT, run `memgpt configure` to set the default backend for MemGPT to point at LM Studio:

```text
# if you are running LM Studio locally, the default IP address + port will be http://localhost:1234
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): lmstudio
? Enter default endpoint: http://localhost:1234
...
```

If you have an existing agent that you want to move to the LM Studio backend, add extra flags to `memgpt run`:

```sh
memgpt run --agent your_agent --model-endpoint-type lmstudio --model-endpoint http://localhost:1234
```
