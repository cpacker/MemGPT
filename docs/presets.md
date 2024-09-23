---
title: Creating new Letta presets
excerpt: Presets allow you to customize agent functionality
category: 6580daaa48aeca0038fc2297
---

Letta **presets** are a combination default settings including a system prompt and a function set. For example, the `letta_docs` preset uses a system prompt that is tuned for document analysis, while the default `memgpt_chat` is tuned for general chatting purposes.

You can create your own presets by creating a `.yaml` file in the `~/.letta/presets` directory. If you want to use a new custom system prompt in your preset, you can create a `.txt` file in the `~/.letta/system_prompts` directory.

For example, if I create a new system prompt and place it in `~/.letta/system_prompts/custom_prompt.txt`, I can then create a preset that uses this system prompt by creating a new file `~/.letta/presets/custom_preset.yaml`:

```yaml
system_prompt: "custom_prompt"
functions:
  - "send_message"
  - "pause_heartbeats"
  - "core_memory_append"
  - "core_memory_replace"
  - "conversation_search"
  - "conversation_search_date"
  - "archival_memory_insert"
  - "archival_memory_search"
```

This preset uses the same base function set as the default presets. You can see the example presets provided [here](https://github.com/cpacker/Letta/tree/main/letta/presets/examples), and you can see example system prompts [here](https://github.com/cpacker/Letta/tree/main/letta/prompts/system).
