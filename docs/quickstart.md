---
title: Quickstart
excerpt: Get up and running with MemGPT
category: 6580d34ee5e4d00068bf2a1d
---

### Installation
> 📘 Using Local LLMs?
>
> If you're using local LLMs refer to the MemGPT + open models page [here](local_llm) for additional installation requirements.

To install MemGPT, make sure you have Python installed on your computer, then run:

```sh
pip install pymemgpt
```

If you are running LLMs locally, you will want to install MemGPT with the local dependencies by running:

```sh
pip install pymemgpt[local]
```

If you already have MemGPT installed, you can update to the latest version with:

```sh
pip install pymemgpt -U
```

### Running MemGPT

Now, you can run MemGPT and start chatting with a MemGPT agent with:

```sh
memgpt run
```

If you're running MemGPT for the first time, you'll see two quickstart options:

1. **OpenAI**: select this if you'd like to run MemGPT with OpenAI models like GPT-4 (requires an OpenAI API key)
2. **MemGPT Free Endpoint**: select this if you'd like to try MemGPT on a top open LLM for free (currently variants of Mixtral 8x7b!)

Neither of these options require you to have an LLM running on your own machine. If you'd like to run MemGPT with your custom LLM setup (or on OpenAI Azure), select **Other** to proceed to the advanced setup.

### Quickstart

If you'd ever like to quickly switch back to the default **OpenAI** or **MemGPT Free Endpoint** options, you can use the `quickstart` command:

```sh
# this will set you up on the MemGPT Free Endpoint
memgpt quickstart
```

```sh
# this will set you up on the default OpenAI settings
memgpt quickstart --backend openai
```

### Advanced setup

MemGPT supports a large number of LLM backends! See:

* [Running MemGPT on OpenAI Azure and custom OpenAI endpoints](endpoints)
* [Running MemGPT with your own LLMs (Llama 2, Mistral 7B, etc.)](local_llm)

### Command-line arguments

The `run` command supports the following optional flags (if set, will override config defaults):

* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--human`: (str) Name of the human to run the agent with.
* `--persona`: (str) Name of agent persona to use.
* `--model`: (str) LLM model to run [gpt-4, gpt-3.5].
* `--preset`: (str) MemGPT preset to run agent with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

### In-chat commands

You can run the following commands during an active chat session in the MemGPT CLI prompt:

* `/exit`: Exit the CLI
* `/attach`: Attach a loaded data source to the agent
* `/save`: Save a checkpoint of the current agent/conversation state
* `/dump`: View the current message log (see the contents of main context)
* `/dump <count>`: View the last <count> messages (all if <count> is omitted)
* `/memory`: Print the current contents of agent memory
* `/pop`: Undo the last message in the conversation
* `/pop <count>`: Undo the last messages in the conversation. It defaults to 3, which usually is one turn around in the conversation
* `/retry`: Pops the last answer and tries to get another one
* `/rethink <text>`: Will replace the inner dialog of the last assistant message with the <text> to help shaping the conversation
* `/rewrite`: Will replace the last assistant answer with the given text to correct or force the answer
* `/heartbeat`: Send a heartbeat system message to the agent
* `/memorywarning`: Send a memory warning system message to the agent

Once you exit the CLI with `/exit`, you can resume chatting with the same agent by specifying the agent name in `memgpt run --agent <NAME>`.

### Examples

Check out the following tutorials on how to set up custom chatbots and chatbots for talking to your data:

* [Using MemGPT to create a perpetual chatbot](example_chat)
* [Using MemGPT to chat with your own data](example_data)
