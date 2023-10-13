# [MemGPT](https://memgpt.ai)


## Setup

Set up dependencies:

```sh
pip install -r requirements.txt
```

Add your OpenAI API key to your environment:

```sh
export OPENAI_API_KEY=YOUR_API_KEY
```

By default MemGPT will use `gpt-4`, so your API key will require `gpt-4` API access.

## MemGPT CLI

To run MemGPT in CLI mode, simply run `main.py`:

```sh
python3 main.py
```

To create a new starter user or starter persona (that MemGPT gets initialized with), create a new `.txt` file in [/memgpt/humans/examples](/memgpt/humans/examples) or [/memgpt/personas](/memgpt/personas/examples), then use the `--persona` or `--human` flag when running `main.py`. For example:

```sh
# assuming you created a new file /memgpt/humans/examples/me.txt
# note: no .txt extension, just 'me'
python main.py --human me
```

### Options

```text
--persona
  load a specific persona file
--human
  load a specific human file
--first
  allows you to send the first message in the chat (by default, MemGPT will send the first message)
```

### CLI commands

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

Debugging commands:

```text
/pop
  undo the last message in the conversation
/heartbeat
  send a heartbeat system message to the agent
/memorywarning
  send a memory warning system message to the agent
```
