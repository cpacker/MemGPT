---
title: Frequently asked questions (FAQ)
excerpt: Check frequently asked questions
category: 6580d34ee5e4d00068bf2a1d
---

> 📘 Open / local LLM FAQ
>
> Questions specific to running your own open / local LLMs with MemGPT can be found [here](local_llm_faq).

## MemGPT CLI

### How can I use MemGPT to chat with my docs?

Check out our [chat with your docs example](example_data) to get started.

### How do I save a chat and continue it later?

When you want to end a chat, run `/exit`, and MemGPT will save your current chat with your agent (make a note of the agent name, e.g. `agent_N`). Later, when you want to start a chat with that same agent, you can run `memgpt run --agent <NAME>`.

### My MemGPT agent is stuck "Thinking..." on the first message?

MemGPT has an extra verification procedure on the very first message to check that in the first message (1) the agent is sending a message to the user, and (2) that the agent is using internal monologue. This verification is meant to avoid the scenario where a bad initial agent message "poisons" the rest of a conversation. For example, a message missing internal monologue might cause all future messages to also omit internal monologue.

If the LLM/model you're using for MemGPT is consistently failing the first message verification, it will appear as a long "Thinking..." loop on the first message. "Weaker" models such as `gpt-3.5-turbo` can frequently fail first message verification because they do not properly use the `send_message` function and instead put the message inside the internal monologue. Better models such as `gpt-4` and `gpt-4-turbo`, as well as open models like `dolphin-2.2.1` and `openhermes-2.5` should not have this problem.

You can disable first message verification by passing the `--no-verify` flag to `memgpt run` (do `memgpt run --no-verify` instead of `memgpt run`). Passing the additional `--debug` flag (`memgpt run --no-verify --debug`) can help you further identify any other issues on first messages that can cause long "Thinking..." loops, such as rate limiting.

### I broke/corrupted my agent, how can I restore an earlier checkpoint?

MemGPT saves agent checkpoints (`.json` files) inside the `~/.memgpt/agents/YOUR_AGENT_NAME/agent_state` directory (`C:\Users\YourUsername\.memgpt\YOUR_AGENT_NAME\agent_state` on Windows). By default, when you load an agent with `memgpt run` it will pull the latest checkpoint `.json` file to load (sorted by date).

If you would like to revert to an earlier checkpoint, if you remove or delete other checkpoint files such that the specific `.json` from the date you would like you use is the most recent checkpoint, then it should get automatically loaded by `memgpt run`. We recommend backing up your agent folder before attempting to delete or remove checkpoint files.

## OpenAI-related

### How do I get an OpenAI key?

To get an OpenAI key, visit [https://platform.openai.com/](https://platform.openai.com/), and make an account.

Then go to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) to create an API key. API keys start with `sk-...`.

### How can I get gpt-4 access?

[https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4)

### I already pay for ChatGPT, is this the same as GPT API?

No, ChatGPT Plus is a separate product from the OpenAI API. Paying for ChatGPT Plus does not get you access to the OpenAI API, vice versa.

### I don't want to use OpenAI, can I still use MemGPT?

Yes, you can run MemGPT with your own LLMs. See our section on local LLMs for information on how to set them up with MemGPT.
