## Quick Start

Install MemGPT:

```sh
pip install pymemgpt
```

Add your OpenAI API key to your environment:

```sh

export OPENAI_API_KEY=YOUR_API_KEY # on Linux/Mac
set OPENAI_API_KEY=YOUR_API_KEY # on Windows
$Env:OPENAI_API_KEY = "YOUR_API_KEY" # on Windows (PowerShell)
```
Configure default setting for MemGPT by running:
```
memgpt configure
```
Now, you can run MemGPT with:
```sh
memgpt run
```
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

You can run the following commands in the MemGPT CLI prompt:

* `/exit`: Exit the CLI
* `/attach`: Attach a loaded data source to the agent
* `/save`: Save a checkpoint of the current agent/conversation state
* `/dump`: View the current message log (see the contents of main context)
* `/memory`: Print the current contents of agent memory
* `/pop`: Undo the last message in the conversation
* `/heartbeat`: Send a heartbeat system message to the agent
* `/memorywarning`: Send a memory warning system message to the agent


Once you exit the CLI with `/exit`, you can resume chatting with the same agent by specifying the agent name in `memgpt run --agent <NAME>`.
