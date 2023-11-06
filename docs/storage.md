# Configuring Storage Backends 
MemGPT supports both local and database storage for archival memory. You can configure which storage backend to use via `memgpt configure`. For larger datasets, we recommend using a database backend. 

!!! warning "Switching storage backends"

    MemGPT can only use one storage backend at a time. If you switch from local to database storage, you will need to re-load data and start agents from scratch. We currently do not support migrating between storage backends.   

## Local
MemGPT will default to using local storage (saved at `~/.memgpt/archival/` for loaded data sources, and `~/.memgpt/agents/` for agent storage). 

## Postgres
In user to us the Postgres backend, you must have a running Postgres database that MemGPT can write to. You can enable the Postgres backend by running `memgpt configure` and selecting `postgres` for archival storage, which will then prompt for the database URI (e.g. `postgresql+pg8000://<USER>:<PASSWORD>@<IP>:5432/<DB_NAME>`)


## Chroma 
(Coming soon)
