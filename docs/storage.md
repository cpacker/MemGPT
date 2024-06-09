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

To run the Postgres backend, you will need a URI to a Postgres database that supports [pgvector](https://github.com/pgvector/pgvector). Follow these steps to set up and run your Postgres server easily with Docker:

1. [Install Docker](https://docs.docker.com/get-docker/)

2. Give the `run_postgres.sh` script permissions to execute:

  ```sh
  chmod +x db/run_postgres.sh
  ```

3. Configure the environment for `pgvector`. You can either:
    - Add the following line to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`):

      ```sh
      export MEMGPT_PGURI=postgresql+pg8000://memgpt:memgpt@localhost:8888/memgpt
      ```

    - Or create a `.env` file in the root project directory with:

      ```sh
      MEMGPT_PGURI=postgresql+pg8000://memgpt:memgpt@localhost:8888/memgpt
      ```

4. Run the script from the root project directory:

  ```sh
  bash db/run_postgres.sh
  ```

5.  Configure MemGPT to use Postgres

```sh
memgpt configure
```

and selecting `postgres` for archival storage, and enter the approporate connection string.  If using docker, change the port in the default value from 5432 to 8888 as shown below.

```text
? Select LLM inference provider: openai
? Override default endpoint: https://api.openai.com/v1
? Select default model (recommended: gpt-4): gpt-4
? Select embedding provider: openai
? Select default preset: memgpt_chat
? Select default persona: sam_pov
? Select default human: cs_phd
? Select storage backend for archival data: postgres
? Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}): postgresql+pg8000://memgpt:memgpt@localhost:8888/memgpt
? Select storage backend for recall data: postgres
? Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}): postgresql+pg8000://memgpt:memgpt@localhost:8888/memgpt
```

Note: You can either use a [hosted provider](https://github.com/pgvector/pgvector/issues/54) or [install pgvector](https://github.com/pgvector/pgvector#installation). You do not need to do this manually if you use our Docker container, however.


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

## Qdrant

To enable the Qdrant backend, make sure to install the required dependencies with:

```sh
pip install 'pymemgpt[qdrant]'
```

You can configure Qdrant with an in-memory instance or a server using the `memgpt configure` command. You can set an API key for authentication with a Qdrant server using the `QDRANT_API_KEY` environment variable. Learn more about setting up Qdrant [here](https://qdrant.tech/documentation/guides/installation/).

```sh
? Select Qdrant backend: server
? Enter the Qdrant instance URI (Default: localhost:6333): localhost:6333
```

## Milvus

To enable the Milvus backend, make sure to install the required dependencies with:

```sh
pip install 'pymemgpt[milvus]'
```
You can configure Milvus connection via command `memgpt configure`.

```sh
...
? Select storage backend for archival data: milvus
? Enter the Milvus connection URI (Default: ~/.memgpt/milvus.db): ~/.memgpt/milvus.db
```
You just set the URI to the local file path, e.g. `~/.memgpt/milvus.db`, which will automatically invoke the local Milvus service instance through Milvus Lite.

If you have large scale of data such as more than a million docs, we recommend setting up a more performant Milvus server on [docker or kubenetes](https://milvus.io/docs/quickstart.md).
And in this case, your URI should be the server URI, e.g. `http://localhost:19530`.
