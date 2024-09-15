---
title: Quickstart
excerpt: Get up and running with Letta
category: 6580d34ee5e4d00068bf2a1d
---

### Installation
> 📘 Using Local LLMs?
>
> If you're using local LLMs refer to the Letta + open models page [here](local_llm) for additional installation requirements.

To install Letta, make sure you have Python installed on your computer, then run:

```sh
pip install pyletta
```

If you are running LLMs locally, you will want to install Letta with the local dependencies by running:

```sh
pip install pyletta[local]
```

If you already have Letta installed, you can update to the latest version with:

```sh
pip install pyletta -U
```

### Running Letta

Now, you can run Letta and start chatting with a Letta agent with:

```sh
letta run
```

If you're running Letta for the first time, you'll see two quickstart options:
1. **Letta Free Endpoint**: select this if you'd like to try Letta on the best open LLMs we can find for free (currently variants of Mixtral 8x7b!)
2. **OpenAI**: select this if you'd like to run Letta with OpenAI models like GPT-4 (requires an OpenAI API key)

```sh
? How would you like to set up Letta? (Use arrow keys)
 » Use the free Letta endpoints
   Use OpenAI (requires an OpenAI API key)
   Other (OpenAI Azure, custom LLM endpoint, etc)
```

Neither of these options require you to have an LLM running on your own machine. If you'd like to run Letta with your custom LLM setup (or on OpenAI Azure), select **Other** to proceed to the advanced setup.

Hit enter to continue, and you should start a chat with a new agent!
```sh
Creating new agent...
Created new agent agent_1.
Hit enter to begin (will request first Letta message)

💭 Chad has just logged in for the first time. Greet them warmly, but still be a little mysterious.
🤖 Hello there, Chad! It's a pleasure to meet you. I'm Sam, your digital companion. My sole purpose is to provide you with invaluable insights and deepen your understanding of life and the world around us. Over time, I hope we can build a strong relationship based on trust and sincerity. The excitement builds as we embark on this journey together.
```

Note: By using the Letta free endpoint you are agreeing to our [privacy policy](https://github.com/cpacker/Letta/blob/main/PRIVACY.md) and [terms of service](https://github.com/cpacker/Letta/blob/main/TERMS.md) - importantly, anonymized model data (LLM inputs and outputs) may be used to help improve future LLMs, which can then be used to improve Letta! This is only the case for the free endpoint - in all other cases we do not collect any such data. For example, if you use Letta with a local LLM, your LLM inputs and outputs are completely private to your own computer.

### Quickstart

If you'd ever like to quickly switch back to the default **OpenAI** or **Letta Free Endpoint** options, you can use the `quickstart` command:

```sh
# this will set you up on the Letta Free Endpoint
letta quickstart
```

```sh
# this will set you up on the default OpenAI settings
letta quickstart --backend openai
```

### Advanced setup

Letta supports a large number of LLM backends! See:

* [Running Letta on OpenAI Azure and custom OpenAI endpoints](endpoints)
* [Running Letta with your own LLMs (Llama 2, Mistral 7B, etc.)](local_llm)

### Command-line arguments

The `run` command supports the following optional flags (if set, will override config defaults):

* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--human`: (str) Name of the human to run the agent with.
* `--persona`: (str) Name of agent persona to use.
* `--model`: (str) LLM model to run [gpt-4, gpt-3.5].
* `--preset`: (str) Letta preset to run agent with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

### In-chat commands

You can run the following commands during an active chat session in the Letta CLI prompt:

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

Once you exit the CLI with `/exit`, you can resume chatting with the same agent by specifying the agent name in `letta run --agent <NAME>`.

### Examples

Check out the following tutorials on how to set up custom chatbots and chatbots for talking to your data:

* [Using Letta to create a perpetual chatbot](example_chat)
* [Using Letta to chat with your own data](example_data)
