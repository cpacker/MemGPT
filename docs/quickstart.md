## Installation

To install MemGPT, make sure you have Python installed on your computer, then run:

```sh
pip install pymemgpt
```

If you already have MemGPT installed, you can update to the latest version with:

```sh
pip install pymemgpt -U --pre
```

### Running MemGPT using the OpenAI API

Add your OpenAI API key to your environment:

```sh
export OPENAI_API_KEY=YOUR_API_KEY # on Linux/Mac
set OPENAI_API_KEY=YOUR_API_KEY # on Windows
$Env:OPENAI_API_KEY = "YOUR_API_KEY" # on Windows (PowerShell)
```
Configure default settings for MemGPT by running:
```sh
memgpt configure
```
Now, you can run MemGPT with:
```sh
memgpt run
```

In this example we use the OpenAI API, but you can run MemGPT with other backends! See:

* [Running MemGPT on OpenAI Azure and custom OpenAI endpoints](endpoints.md)
* [Running MemGPT with your own LLMs (Llama 2, Mistral 7B, etc.)](local_llm.md)

### In-chat commands

You can run the following commands during an active chat session in the MemGPT CLI prompt:

* `/exit`: Exit the CLI
* `/attach`: Attach a loaded data source to the agent
* `/save`: Save a checkpoint of the current agent/conversation state
* `/dump`: View the current message log (see the contents of main context)
* `/memory`: Print the current contents of agent memory
* `/pop`: Undo the last message in the conversation
* `/heartbeat`: Send a heartbeat system message to the agent
* `/memorywarning`: Send a memory warning system message to the agent

Once you exit the CLI with `/exit`, you can resume chatting with the same agent by specifying the agent name in `memgpt run --agent <NAME>`.

### Examples

Check out the following tutorials on how to set up custom chatbots and chatbots for talking to your data:

* [Using MemGPT to create a perpetual chatbot](example_chat.md)
* [Using MemGPT to chat with your own data](example_data.md)
