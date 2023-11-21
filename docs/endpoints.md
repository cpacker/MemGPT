### LLM Backends

You can use MemGPT with various LLM backends, including the OpenAI API, Azure OpenAI, and various local (or self-hosted) LLM backends.

#### OpenAI
To use MemGPT with an OpenAI API key, simply set the `OPENAI_API_KEY` variable:
```sh
export OPENAI_API_KEY=YOUR_API_KEY # on Linux/Mac
set OPENAI_API_KEY=YOUR_API_KEY # on Windows
$Env:OPENAI_API_KEY = "YOUR_API_KEY" # on Windows (PowerShell)
```

#### Azure
To use MemGPT with Azure, expore the following variables and then re-run `memgpt configure`:
```sh
# see https://github.com/openai/openai-python#microsoft-azure-endpoints
export AZURE_OPENAI_KEY = ...
export AZURE_OPENAI_ENDPOINT = ...
export AZURE_OPENAI_VERSION = ...

# set the below if you are using deployment ids
export AZURE_OPENAI_DEPLOYMENT = ...
export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = ...
```

Replace `export` with `set` or `$Env:` if you are on Windows (see the OpenAI example).

Note: **your Azure endpoint must support functions** or you will get an error. See [this GitHub issue](https://github.com/cpacker/MemGPT/issues/91) for more information.

#### Custom endpoints
To use custom OpenAI endpoints, run `export OPENAI_API_BASE=<MY_CUSTOM_URL>` and then re-run `memgpt configure` to set the custom endpoint as the default endpoint.

#### Local LLMs
Setting up MemGPT to run with local LLMs requires a bit more setup, follow [the instructions here](../local_llm).
