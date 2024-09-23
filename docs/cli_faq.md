---
title: Frequently asked questions (FAQ)
excerpt: Check frequently asked questions
category: 6580d34ee5e4d00068bf2a1d
---

> 📘 Open / local LLM FAQ
>
> Questions specific to running your own open / local LLMs with Letta can be found [here](local_llm_faq).

## Letta CLI

### How can I use Letta to chat with my docs?

Check out our [chat with your docs example](example_data) to get started.

### How do I save a chat and continue it later?

When you want to end a chat, run `/exit`, and Letta will save your current chat with your agent (make a note of the agent name, e.g. `agent_N`). Later, when you want to start a chat with that same agent, you can run `letta run --agent <NAME>`.

### How do I implement Letta for multiple users?
The REST API for [Letta](https://letta.readme.io/reference/api) is flexible and leverages PostgreSQL DB or SQLite for its backend. To implement a multi-user setup, first determine the user_id (either create a UUID or use the user_id from your own database). Then [create an agent](https://letta.readme.io/reference/create_agent_api_agents_post), and finally use the agent_id and user_id to post a message or run a command. Internally the following occurs: 
* a user creates an agent
* that agent is "owned" by a user
* when the user sends the agent a message, that's stored in a message collection (messages are indexed by user and agent ids)
* on the higher-level agents side (not talking about db implementation details), the agent can only see a few messages at a time, but has access to all the messages ever sent between it and the user via the recall memory search functions
* the database is multi-user, and the REST endpoints function in a way where user data is not shared

### My Letta agent is stuck "Thinking..." on the first message?

Letta has an extra verification procedure on the very first message to check that in the first message (1) the agent is sending a message to the user, and (2) that the agent is using internal monologue. This verification is meant to avoid the scenario where a bad initial agent message "poisons" the rest of a conversation. For example, a message missing internal monologue might cause all future messages to also omit internal monologue.

If the LLM/model you're using for Letta is consistently failing the first message verification, it will appear as a long "Thinking..." loop on the first message. "Weaker" models such as `gpt-3.5-turbo` can frequently fail first message verification because they do not properly use the `send_message` function and instead put the message inside the internal monologue. Better models such as `gpt-4` and `gpt-4-turbo`, as well as open models like `dolphin-2.2.1` and `openhermes-2.5` should not have this problem.

You can disable first message verification by passing the `--no-verify` flag to `letta run` (do `letta run --no-verify` instead of `letta run`). Passing the additional `--debug` flag (`letta run --no-verify --debug`) can help you further identify any other issues on first messages that can cause long "Thinking..." loops, such as rate limiting.

### What are personas and how they relate to agents and humans? 
Letta has two core components: agents and humans. Each human contains information about the user that is continously updated as Letta learns more about that user. Agents are what the human interacts with when they chat with Letta. Each agent can be customized through presets which are basically the configuration for an agent and includes the following componenets:
* system prompt (you usually don't change this)
* persona (personality of your bot and their initial memories)
* human (description of yourself / user details)
* functions (the functions the agent can call during convo)

### I broke/corrupted my agent, how can I restore an earlier checkpoint?

Letta saves agent checkpoints (`.json` files) inside the `~/.letta/agents/YOUR_AGENT_NAME/agent_state` directory (`C:\Users\YourUsername\.letta\YOUR_AGENT_NAME\agent_state` on Windows). By default, when you load an agent with `letta run` it will pull the latest checkpoint `.json` file to load (sorted by date).

If you would like to revert to an earlier checkpoint, if you remove or delete other checkpoint files such that the specific `.json` from the date you would like you use is the most recent checkpoint, then it should get automatically loaded by `letta run`. We recommend backing up your agent folder before attempting to delete or remove checkpoint files.

## OpenAI-related

### How do I get an OpenAI key?

To get an OpenAI key, visit [https://platform.openai.com/](https://platform.openai.com/), and make an account.

Then go to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys) to create an API key. API keys start with `sk-...`.

### How can I get gpt-4 access?

[https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4)

### I already pay for ChatGPT, is this the same as GPT API?

No, ChatGPT Plus is a separate product from the OpenAI API. Paying for ChatGPT Plus does not get you access to the OpenAI API, vice versa.

### I don't want to use OpenAI, can I still use Letta?

Yes, you can run Letta with your own LLMs. See our section on local LLMs for information on how to set them up with Letta.
