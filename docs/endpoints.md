### LLM Backends

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

Note: your Azure endpoint must support functions or you will get an error. See https://github.com/cpacker/MemGPT/issues/91 for more information.

#### Custom Endpoints
To use custom endpoints, run `export OPENAI_API_BASE=<MY_CUSTOM_URL>` and then re-run `memgpt configure` to set the custom endpoint as the default endpoint.

#### Local LLMs


TODO: Link to full documentation on configuring open models
