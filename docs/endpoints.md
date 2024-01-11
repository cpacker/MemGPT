---
title: Configuring LLM backends
excerpt: Connecting MemGPT to various LLM backends
category: 6580d34ee5e4d00068bf2a1d
---

You can use MemGPT with various LLM backends, including the OpenAI API, Azure OpenAI, and various local (or self-hosted) LLM backends.

## OpenAI

To use MemGPT with an OpenAI API key, simply set the `OPENAI_API_KEY` variable:

```sh
export OPENAI_API_KEY=YOUR_API_KEY # on Linux/Mac
set OPENAI_API_KEY=YOUR_API_KEY # on Windows
$Env:OPENAI_API_KEY = "YOUR_API_KEY" # on Windows (PowerShell)
```

When you run `memgpt configure`, make sure to select `openai` for both the LLM inference provider and embedding provider, for example:

```text
$ memgpt configure
? Select LLM inference provider: openai
? Override default endpoint: https://api.openai.com/v1
? Select default model (recommended: gpt-4): gpt-4
? Select embedding provider: openai
? Select default preset: memgpt_chat
? Select default persona: sam_pov
? Select default human: cs_phd
? Select storage backend for archival data: local
```

### OpenAI Proxies

To use custom OpenAI endpoints, specify a proxy URL when running `memgpt configure` to set the custom endpoint as the default endpoint.

## Azure OpenAI

To use MemGPT with Azure, expore the following variables and then re-run `memgpt configure`:

```sh
# see https://github.com/openai/openai-python#microsoft-azure-endpoints
export AZURE_OPENAI_KEY=...
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_VERSION=...

# set the below if you are using deployment ids
export AZURE_OPENAI_DEPLOYMENT=...
export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=...
```

For example, if your endpoint is `customproject.openai.azure.com` (for both your GPT model and your embeddings model), you would set the following:

```sh
# change AZURE_OPENAI_VERSION to the latest version
export AZURE_OPENAI_KEY="YOUR_AZURE_KEY"
export AZURE_OPENAI_VERSION="2023-08-01-preview"
export AZURE_OPENAI_ENDPOINT="https://customproject.openai.azure.com"
export AZURE_OPENAI_EMBEDDING_ENDPOINT="https://customproject.openai.azure.com"
```

If you named your deployments names other than their defaults, you would also set the following:

```sh
# assume you called the gpt-4 (1106-Preview) deployment "personal-gpt-4-turbo"
export AZURE_OPENAI_DEPLOYMENT="personal-gpt-4-turbo"

# assume you called the text-embedding-ada-002 deployment "personal-embeddings"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="personal-embeddings"
```

Replace `export` with `set` or `$Env:` if you are on Windows (see the OpenAI example).

When you run `memgpt configure`, make sure to select `azure` for both the LLM inference provider and embedding provider, for example:

```text
$ memgpt configure
? Select LLM inference provider: azure
? Select default model (recommended: gpt-4): gpt-4-1106-preview
? Select embedding provider: azure
? Select default preset: memgpt_chat
? Select default persona: sam_pov
? Select default human: cs_phd
? Select storage backend for archival data: local
```

Note: **your Azure endpoint must support functions** or you will get an error. See [this GitHub issue](https://github.com/cpacker/MemGPT/issues/91) for more information.

## Local Models & Custom Endpoints

MemGPT supports running open source models, both being run locally or as a hosted service. Setting up MemGPT to run with open models requires a bit more setup, follow [the instructions here](local_llm).
