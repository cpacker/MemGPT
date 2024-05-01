---
title: Attaching data sources
excerpt: Connecting external data to your MemGPT agent
category: 6580d34ee5e4d00068bf2a1d
---

MemGPT supports pre-loading data into archival memory. In order to made data accessible to your agent, you must load data in with `memgpt load`, then attach the data source to your agent. You can configure where archival memory is stored by configuring the [storage backend](storage).

### Viewing available data sources

You can view available data sources with:

```sh CLI
memgpt list sources
```
```python Python
from memgpt import create_client

# Connect to the server as a user
client = create_client()

# List data source names that belong to user
client.list_sources()
```

```sh
+----------------+----------+----------+
|      Name      | Location | Agents   |
+----------------+----------+----------+
| short-stories  |  local   |  agent_1 |
|      arxiv     |  local   |          |
|  memgpt-docs   |  local   |  agent_1 |
+----------------+----------+----------+
```

The `Agents` column indicates which agents have access to the data, while `Location` indicates what storage backend the data has been loaded into.

### Attaching data to agents

Attaching a data source to your agent loads the data into your agent's archival memory to access. 


```sh CLI
memgpt run 
...
> Enter your message: /attach
? Select data source (Use arrow keys)
 » short-stories
   arxiv
   memgpt-docs
```
```python Python
from memgpt import create_client

# Connect to the server as a user
client = create_client()

# Create an agent 
agent = client.create_agent()

# Attach a source to an agent 
client.attach_source_to_agent(source_name="short-storie", agent_id=agent.id)
```

> 👍 Hint
> To encourage your agent to reference its archival memory, we recommend adding phrases like "_search your archival memory..._" for the best results.

### Loading a file or directory

You can load a file, list of files, or directly into MemGPT with the following command:

```sh
memgpt load directory --name <NAME> \
    [--input-dir <DIRECTORY>] [--input-files <FILE1> <FILE2>...] [--recursive]
```
```python Python
from memgpt import create_client

# Connect to the server as a user
client = create_client()

# Create a data source 
source = client.create_source(name="example_source")

# Add file data into a source 
client.load_file_into_source(filename=filename, source_id=source.id)
```

### Loading with custom connectors 
You can implement your own data connectors in MemGPT, and use them to load data into data sources: 

```python Python
from memgpt.data_sources.connectors import DataConnector

class DummyDataConnector(DataConnector):
    """Fake data connector for texting which yields document/passage texts from a provided list"""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def generate_documents(self) -> Iterator[Tuple[str, Dict]]:
        for text in self.texts:
            yield text, {"metadata": "dummy"}

    def generate_passages(self, documents: List[Document], chunk_size: int = 1024) -> Iterator[Tuple[str | Dict]]:
        for doc in documents:
            yield doc.text, doc.metadata
```
