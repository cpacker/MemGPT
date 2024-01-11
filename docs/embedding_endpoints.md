---
title: Configuring embedding backends
excerpt: Connecting MemGPT to various endpoint backends
category: 6580d34ee5e4d00068bf2a1d
---

MemGPT uses embedding models for retrieval search over archival memory. You can use embeddings provided by OpenAI, Azure, or any model on Hugging Face.

## OpenAI

To use OpenAI, make sure your `OPENAI_API_KEY` environment variable is set.

```sh
export OPENAI_API_KEY=YOUR_API_KEY # on Linux/Mac
```

Then, configure MemGPT and select `openai` as the embedding provider:

```text
> memgpt configure
...
? Select embedding provider: openai
...
```

## Azure

To use Azure, set environment variables for Azure and an additional variable specifying your embedding deployment:

```sh
# see https://github.com/openai/openai-python#microsoft-azure-endpoints
export AZURE_OPENAI_KEY = ...
export AZURE_OPENAI_ENDPOINT = ...
export AZURE_OPENAI_VERSION = ...

# set the below if you are using deployment ids
export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = ...
```

Then, configure MemGPT and select `azure` as the embedding provider:

```text
> memgpt configure
...
? Select embedding provider: azure
...
```

## Custom Endpoint

MemGPT supports running embeddings with any Hugging Face model using the [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)(TEI) library. To get started, first make sure you follow TEI's [instructions](https://github.com/huggingface/text-embeddings-inference#get-started) for getting started. Once you have a running endpoint, you can configure MemGPT to use your endpoint:

```text
> memgpt configure
...
? Select embedding provider: hugging-face
? Enter default endpoint: http://localhost:8080
? Enter HuggingFace model tag (e.g. BAAI/bge-large-en-v1.5): BAAI/bge-large-en-v1.5
? Enter embedding model dimentions (e.g. 1024): 1536
...
```

## Local Embeddings

MemGPT can compute embeddings locally using a lightweight embedding model [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5).

> 🚧 Local LLM Performance
>
> The `BAAI/bge-small-en-v1.5` was chosen to be lightweight, so you may notice degraded performance with embedding-based retrieval when using this option.

To compute embeddings locally, install dependencies with:

```sh
pip install `pymemgpt[local]`
```

Then, select the `local` option during configuration:

```text
memgpt configure

...
? Select embedding provider: local
...
```
