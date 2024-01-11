---
title: Configuring storage backends
excerpt: Customizing the MemGPT storage backend
category: 6580d34ee5e4d00068bf2a1d
---

> ⚠️ Switching storage backends
>
> MemGPT can only use one storage backend at a time. If you switch from local to database storage, you will need to re-load data and start agents from scratch. We currently do not support migrating between storage backends.

MemGPT supports both local and database storage for archival memory. You can configure which storage backend to use via `memgpt configure`. For larger datasets, we recommend using a database backend.

## Local

MemGPT will default to using local storage (saved at `~/.memgpt/archival/` for loaded data sources, and `~/.memgpt/agents/` for agent storage).

## Postgres

In order to use the Postgres backend, you must have a running Postgres database that MemGPT can write to. You can enable the Postgres backend by running `memgpt configure` and selecting `postgres` for archival storage, which will then prompt for the database URI (e.g. `postgresql+pg8000://<USER>:<PASSWORD>@<IP>:5432/<DB_NAME>`). To enable the Postgres backend, make sure to install the required dependencies with:

```sh
pip install 'pymemgpt[postgres]'
```

### Running Postgres

You will need to have a URI to a Postgres database which support [pgvector](https://github.com/pgvector/pgvector). You can either use a [hosted provider](https://github.com/pgvector/pgvector/issues/54) or [install pgvector](https://github.com/pgvector/pgvector#installation).

## Chroma

You can configure Chroma with both the HTTP and persistent storage client via `memgpt configure`. You will need to specify either a persistent storage path or host/port dependending on your client choice. The example below shows how to configure Chroma with local persistent storage:

```text
? Select LLM inference provider: openai
? Override default endpoint: https://api.openai.com/v1
? Select default model (recommended: gpt-4): gpt-4
? Select embedding provider: openai
? Select default preset: memgpt_chat
? Select default persona: sam_pov
? Select default human: cs_phd
? Select storage backend for archival data: chroma
? Select chroma backend: persistent
? Enter persistent storage location: /Users/sarahwooders/.memgpt/config/chroma
```

## LanceDB

You have to enable the LanceDB backend by running

```sh
memgpt configure
```

and selecting `lancedb` for archival storage, and database URI (e.g. `./.lancedb`"), Empty archival uri is also handled and default uri is set at `./.lancedb`. For more checkout [lancedb docs](https://lancedb.github.io/lancedb/)
