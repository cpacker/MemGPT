# Configuring Storage Backends

!!! warning "Switching storage backends"

    MemGPT can only use one storage backend at a time. If you switch from local to database storage, you will need to re-load data and start agents from scratch. We currently do not support migrating between storage backends.

MemGPT supports both local and database storage for archival memory. You can configure which storage backend to use via `memgpt configure`. For larger datasets, we recommend using a database backend.

## Local
MemGPT will default to using local storage (saved at `~/.memgpt/archival/` for loaded data sources, and `~/.memgpt/agents/` for agent storage).

## Postgres
In order to use the Postgres backend, you must have a running Postgres database that MemGPT can write to. You can enable the Postgres backend by running `memgpt configure` and selecting `postgres` for archival storage, which will then prompt for the database URI (e.g. `postgresql+pg8000://<USER>:<PASSWORD>@<IP>:5432/<DB_NAME>`). To enable the Postgres backend, make sure to install the required dependencies with:
```
pip install 'pymemgpt[postgres]'
```

### Running Postgres
You will need to have a URI to a Postgres database which support [pgvector](https://github.com/pgvector/pgvector). You can either use a [hosted provider](https://github.com/pgvector/pgvector/issues/54) or [install pgvector](https://github.com/pgvector/pgvector#installation).


## LanceDB
In order to use the LanceDB backend.

 You have to enable the LanceDB backend by running 
 
 ```
 memgpt configure
 ```
  and selecting `lancedb` for archival storage, and database URI (e.g. `./.lancedb`"), Empty archival uri is also handled and default uri is set at `./.lancedb`. 

To enable the LanceDB backend, make sure to install the required dependencies with:
```
pip install 'pymemgpt[lancedb]'
```
for more checkout [lancedb docs](https://lancedb.github.io/lancedb/)


## Chroma
(Coming soon)
