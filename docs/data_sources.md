## Loading External Data
MemGPT supports pre-loading data into archival memory. In order to made data accessible to your agent, you must:

1. Load the data into MemGPT as a *data source*.

2. Attach the data source to your agent, which will load it into archival memory.

### Loading External Data

We currently support loading from a directory and database dumps. We highly encourage contributions for new data sources, which can be added as a new [CLI data load command](https://github.com/cpacker/MemGPT/blob/main/memgpt/cli/cli_load.py).

Loading from a directory:
```sh
# loading a directory
memgpt load directory --name <NAME> \
    [--input-dir <DIRECTORY>] [--input-files <FILE1> <FILE2>...] [--recursive]
```
Loading from a database dump:
```sh
memgpt load database --name <NAME>  \
    --query <QUERY> \ # Query to run on database to get data
    --dump-path <PATH> \ # Path to dump file
    --scheme <SCHEME> \ # Database scheme
    --host <HOST> \ # Database host
    --port <PORT> \ # Database port
    --user <USER> \ # Database user
    --password <PASSWORD> \ # Database password
    --dbname <DB_NAME> # Database name
```
To encourage your agent to reference its archival memory, we recommend adding phrases like "_search your archival memory..._" for the best results.


### Connecting data to an agent

You can attach data to your agent (which will place the data in your agent's archival memory) in two ways:

1. Run `memgpt attach --agent <AGENT-NAME> --data-source <DATA-SOURCE-NAME>`
2. While chatting with the agent, enter the `/attach` command and select the data source
