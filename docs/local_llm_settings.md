---
title: Customizing LLM parameters
excerpt: How to set LLM inference parameters (advanced)
category: 6580da9a40bb410016b8b0c3
---

> 📘 Understanding different parameters
>
> The [llama.cpp docs](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md) have a great breakdown explaining the effect of modifying different parameters.

By default, MemGPT will specify the minimum necessary parameters when communicating with the LLM backend. This includes parameters such as context length and the prompt itself, but does not include other important parameters such as temperature.

This means that many LLM inference parameters (such as temperature) will be set to their defaults specified by the LLM backend you are using, so if two different backends have very different default parameters, MemGPT may perform very differently on the two backends even when using the exact same LLM on both.

## Customizing your LLM parameters in the settings file

### Finding the settings file

To set your own parameters passed to custom LLM backends (ie non-OpenAI endpoints), you can modify the file `completions_api_settings.json` located in your MemGPT home folder.

On Linux/MacOS, the file will be located at:

```sh
~/.memgpt/settings/completions_api_settings.json
```

And on Windows:

```batch
C:\Users\[YourUsername]\.memgpt\settings\completions_api_settings.json
```

You can also use the `memgpt folder` command which will open the home directory for you:

```sh
# this should pop open a folder view on your system
memgpt folder
```

### Customizing the settings file

Once you've found the file, you can open it your text editor of choice and add fields to the JSON that correspond to parameters in your particular LLM backend. The JSON file itself will be empty (indicating no user-specified settings), and any settings you add to the file will be passed through to the LLM backend.

When editing the file, make sure you are using parameters that are specified by the backend API you're using. In many cases, the naming scheme will follow the [llama.cpp conventions](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md) or the [OpenAI Completions API conventions](https://platform.openai.com/docs/api-reference/completions/create), but make sure to check the documentation of the specific backend you are using. **If parameters are misspecified it may cause your LLM backend to throw an error or crash.**

Additionally, make sure that your settings file is valid JSON. Many text editors will highlight invalid JSON, but you can also check your JSON using [tools online](https://jsonformatter.org/).

### Example: LM Studio (simple)

As a simple example, let's try setting the temperature. Assuming we've already [set up LM Studio](lmstudio), if we start a MemGPT chat while using the LM Studio API, we'll see the request and it's associated parameters inside the LM Studio server logs, and it contains `"temp": 0.8`:

```sh
[INFO] Provided inference configuration: {
  ...(truncated)...
  "temp": 0.8,
  ...(truncated)...
}
```

Let's try changing the temperature to `1.0`. In our `completions_api_settings.json` file, we set the following:

```json
{
    "temperature": 1.0
}
```

Note how we're using the naming conventions from llama.cpp. In this case, using `"temperature"` instead of `"temp"`.

Now if we save the file and start a new agent chat with `memgpt run`, we'll notice that the LM Studio server logs now say `"temp": 1.0`:

```sh
[INFO] Provided inference configuration: {
  ...(truncated)...
  "temp": 1,
  ...(truncated)...
}
```

Hooray! That's the gist of it - simply set parameters in your JSON file and they will be passed through to the LLM backend.

### Checking that your settings are being loaded

With LM Studio we can observe the settings that are loaded in the server logs, but with some backends you may not be able to see the parameters of the request so it can be difficult to tell if your settings file is getting loaded correctly.

To double-check that your settings are being loaded and passed to the backend, you can run MemGPT with the `--debug` parameter and look for the relevant output:

```sh
memgpt run --debug
```

If your parameters are getting picked up correctly, they will be output to the terminal:

```sh
...(truncated)...
Found completion settings file '/Users/user/.memgpt/settings/completions_api_settings.json', loading it...
Updating base settings with the following user settings:
{
  "temperature": 1.0
}
...(truncated)...
```

If you have an empty settings file or your file wasn't saved properly, you'll see the following message:

```sh
...(truncated)...
Found completion settings file '/Users/loaner/.memgpt/settings/completions_api_settings.json', loading it...
'/Users/user/.memgpt/settings/completions_api_settings.json' was empty, ignoring...
...(truncated)...
```

### Example: LM Studio (advanced)

In practice, there are many parameters you might want to set, since tuning these parameters can dramatically alter the tone or feel of the generated LLM outputs. Let's try changing a larger set of parameters.

Now just for reference, let's record the set of parameters before any modifications (truncated to include the parameters we're changing only):

```text
[INFO] Provided inference configuration: {
  ...(truncated)...
  "top_k": 40,
  "top_p": 0.95,
  "temp": 1,
  "repeat_penalty": 1.1,
  "seed": -1,
  "tfs_z": 1,
  "typical_p": 1,
  "repeat_last_n": 64,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "mirostat": 0,
  "mirostat_tau": 5,
  "mirostat_eta": 0.1,
  "penalize_nl": true,
  ...(truncated)...
}
```

Now copy the following to your `completions_api_settings.json` file:

```json
{
    "top_k": 1,
    "top_p": 0,
    "temperature": 0,
    "repeat_penalty": 1.18,
    "seed": -1,
    "tfs_z": 1,
    "typical_p": 1,
    "repeat_last_n": 64,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "mirostat": 2,
    "mirostat_tau": 4,
    "mirostat_eta": 0.1,
    "penalize_nl": false
}
```

When we run, our settings are updated:

```text
[INFO] Provided inference configuration: {
  ...(truncated)...
  "top_k": 1,
  "top_p": 0,
  "temp": 0,
  "repeat_penalty": 1.18,
  "seed": -1,
  "tfs_z": 1,
  "typical_p": 1,
  "repeat_last_n": 64,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "mirostat": 0,
  "mirostat_tau": 5,
  "mirostat_eta": 0.1,
  "penalize_nl": true,
  ...(truncated)...
}
```
