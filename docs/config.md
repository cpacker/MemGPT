### Configuring the agent
You can set agent defaults by running `memgpt configure`.

The `memgpt run` command supports the following optional flags (if set, will override config defaults):

* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--human`: (str) Name of the human to run the agent with.
* `--persona`: (str) Name of agent persona to use.
* `--model`: (str) LLM model to run [gpt-4, gpt-3.5].
* `--preset`: (str) MemGPT preset to run agent with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)


### Adding Custom Personas/Humans
You can add new human or persona definitions either by providing a file (using the `-f` flag) or text (using the `--text` flag).
```sh
# add a human
memgpt add human [--name <NAME>] [-f <FILENAME>] [--text <TEXT>]

# add a persona
memgpt add persona [--name <NAME>] [-f <FILENAME>] [--text <TEXT>]
```

You can view available persona and human files with the following command:
```sh
memgpt list [humans/personas]
```

### Custom Presets
