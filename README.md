<div align="center">
  
# [MemGPT](https://memgpt.ai)

[![Discord](https://img.shields.io/discord/1111172801899012102?label=Discord&logo=discord&logoColor=green&style=flat-square)](https://discord.gg/9GEQrxmVyE)

:rocket: Teaching LLMs memory management for unbounded context :rocket:

</div>

## Quick Setup

Set up dependencies:

```sh
pip install -r requirements.txt
```

Add your OpenAI API key to your environment:

```sh
export OPENAI_API_KEY=YOUR_API_KEY
```

## What is MemGPT? 

MemoryGPT (or MemGPT in short) is a system that intelligently manages different memory tiers in LLMs in order to effectively provide extended context within the LLM's limited context window. For example, MemGPT knows when to push critical information to a vector database and when to retrieve it later in the chat for perpetual conversations. Learn more about MemGPT in our [paper](https://arxiv.org/abs/2310.08560). 

## Try MemGPT in your CLI

To run MemGPT for as a conversation agent in CLI mode, simply run `main.py`:

```sh
python3 main.py
```

To create a new starter user or starter persona (that MemGPT gets initialized with), create a new `.txt` file in [/memgpt/humans/examples](/memgpt/humans/examples) or [/memgpt/personas](/memgpt/personas/examples), then use the `--persona` or `--human` flag when running `main.py`. For example:

```sh
# assuming you created a new file /memgpt/humans/examples/me.txt
python main.py --human me
```

### `main.py` flags

```text
--persona
  load a specific persona file
--human
  load a specific human file
--first
  allows you to send the first message in the chat (by default, MemGPT will send the first message)
```

### Interactive CLI commands

While using MemGPT via the CLI you can run various commands

Basic commands:

```text
/save
  save a checkpoint of the current agent/conversation state
/load
  load a saved checkpoint
/dump
  view the current message log (see the contents of main context)
/memory
  print the current contents of agent memory
```

### Support

* For CLI, no `.txt` extension, just `me` for `--human` flag
* By default MemGPT will use `gpt-4`, so your API key will require `gpt-4` API access.
* If you would like to further debug MemGPT, we provide the further options: 

  ```text
  /pop
    undo the last message in the conversation
  /heartbeat
    send a heartbeat system message to the agent
  /memorywarning
    send a memory warning system message to the agent
  ```

If you have any further questions, or have anything to share, we are excited to hear your feedback!

* For issues and feature requests, please [open a GitHub issue](https://github.com/cpacker/MemGPT/issues).
