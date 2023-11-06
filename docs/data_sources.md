## Loading External Data
MemGPT supports pre-loading data into archival memory. In order to made data accessible to your agent, you must load data in with `memgpt load`, then attach the data source to your agent. You can view available data sources with: 
```
memgpt list sources
```
You can attach data to your agent (which will place the data in your agent's archival memory) in two ways:

1. Run `memgpt attach --agent <AGENT-NAME> --data-source <DATA-SOURCE-NAME>`
2. While chatting with the agent, enter the `/attach` command and select the data source

To encourage your agent to reference its archival memory, we recommend adding phrases like "_search your archival memory..._" for the best results.



### Loading a file or directory
You can load a file, list of files, or directry into MemGPT with the following command: 
```sh
memgpt load directory --name <NAME> \
    [--input-dir <DIRECTORY>] [--input-files <FILE1> <FILE2>...] [--recursive]
```


### Loading a database dump 
You can load database into MemGPT, either from a database dump or a database connection, with the following command: 
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

### Loading a vector database 
If you already have a vector database containing passages and embeddings, you can load them into MemGPT by specifying the table name, database URI, and the columns containing the passage text and embeddings.  
```sh
memgpt load vector-database --name <NAME> \
    --uri <URI> \ # Database URI
    --table_name <TABLE-NAME> \ # Name of table containing data 
    --text_column <TEXT-COL> \ # Name of column containing text
    --embedding_column <EMBEDDING-COL> # Name of column containing embedding
```
Since embeddings are already provided, MemGPT will not re-compute the embeddings. 

### Loading a Llama Index dump 
If you have a Llama Index `VectorIndex` which was saved to disk, you can load it into MemGPT by specifying the directory the index was saved to: 
```sh
memgpt load index --name <NAME> --dir <INDEX-DIR>
```
Since Llama Index will have already computing embeddings, MemGPT will not re-compute embeddings. 


### Loading custom data sources
We highly encourage contributions for new data sources, which can be added as a new [CLI data load command](https://github.com/cpacker/MemGPT/blob/main/memgpt/cli/cli_load.py).
