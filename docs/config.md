---
title: Configuration
excerpt: Configuring your Letta agent
category: 6580d34ee5e4d00068bf2a1d
---

You can set agent defaults by running `letta configure`, which will store config information at `~/.letta/config` by default.

The `letta run` command supports the following optional flags (if set, will override config defaults):

* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--human`: (str) Name of the human to run the agent with.
* `--persona`: (str) Name of agent persona to use.
* `--model`: (str) LLM model to run (e.g. `gpt-4`, `dolphin_xxx`)
* `--preset`: (str) Letta preset to run agent with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

You can override the parameters you set with `letta configure` with the following additional flags specific to local LLMs:

* `--model-wrapper`: (str) Model wrapper used by backend (e.g. `airoboros_xxx`)
* `--model-endpoint-type`: (str) Model endpoint backend type (e.g. lmstudio, ollama)
* `--model-endpoint`: (str) Model endpoint url (e.g. `localhost:5000`)
* `--context-window`: (int) Size of model context window (specific to model type)

#### Updating the config location

You can override the location of the config path by setting the environment variable `MEMGPT_CONFIG_PATH`:

```sh
export MEMGPT_CONFIG_PATH=/my/custom/path/config # make sure this is a file, not a directory
```

### Adding Custom Personas/Humans

You can add new human or persona definitions either by providing a file (using the `-f` flag) or text (using the `--text` flag).

```sh
# add a human
letta add human [--name <NAME>] [-f <FILENAME>] [--text <TEXT>]

# add a persona
letta add persona [--name <NAME>] [-f <FILENAME>] [--text <TEXT>]
```

You can view available persona and human files with the following command:

```sh
letta list [humans/personas]
```

### Custom Presets

You can customize your Letta agent even further with [custom presets](presets) and [custom functions](functions).
