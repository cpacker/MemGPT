<a href="#user-content-memgpt"><img src="https://memgpt.ai/assets/img/memgpt_logo_circle.png" alt="MemGPT logo" width="75" align="right"></a>

# [MemGPT](https://memgpt.ai)

<div align="center">

 <strong>Try out our MemGPT chatbot on <a href="https://discord.gg/9GEQrxmVyE">Discord</a>!</strong>

 <strong>⭐ NEW: You can now run MemGPT with <a href="https://github.com/cpacker/MemGPT/discussions/67">local LLMs</a> and <a href="https://github.com/cpacker/MemGPT/discussions/65">AutoGen</a>! ⭐ </strong>

[![Discord](https://img.shields.io/discord/1161736243340640419?label=Discord&logo=discord&logoColor=5865F2&style=flat-square&color=5865F2)](https://discord.gg/9GEQrxmVyE)
[![arXiv 2310.08560](https://img.shields.io/badge/arXiv-2310.08560-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2310.08560)

</div>

<details open>
  <summary><h2>🤖 Create perpetual chatbots with self-editing memory!</h2></summary>
  <div align="center">
    <br>
    <img src="https://memgpt.ai/assets/img/demo.gif" alt="MemGPT demo video" width="800">
  </div>
</details>

<details>
 <summary><h2>🗃️ Chat with your data - talk to your local files or SQL database!</strong></h2></summary>
  <div align="center">
    <img src="https://memgpt.ai/assets/img/doc.gif" alt="MemGPT demo video for sql search" width="800">
  </div>
</details>

## Quick setup

Join <a href="https://discord.gg/9GEQrxmVyE">Discord</a></strong> and message the MemGPT bot (in the `#memgpt` channel). Then run the following commands (messaged to "MemGPT Bot"):
* `/profile` (to create your profile)
* `/key` (to enter your OpenAI key)
* `/create` (to create a MemGPT chatbot)

Make sure your privacy settings on this server are open so that MemGPT Bot can DM you: \
MemGPT → Privacy Settings → Direct Messages set to ON
<div align="center">
 <img src="https://memgpt.ai/assets/img/discord/dm_settings.png" alt="set DMs settings on MemGPT server to be open in MemGPT so that MemGPT Bot can message you" width="400">
</div>

You can see the full list of available commands when you enter `/` into the message box.
<div align="center">
 <img src="https://memgpt.ai/assets/img/discord/slash_commands.png" alt="MemGPT Bot slash commands" width="400">
</div>

## What is MemGPT?

Memory-GPT (or MemGPT in short) is a system that intelligently manages different memory tiers in LLMs in order to effectively provide extended context within the LLM's limited context window. For example, MemGPT knows when to push critical information to a vector database and when to retrieve it later in the chat, enabling perpetual conversations. Learn more about MemGPT in our [paper](https://arxiv.org/abs/2310.08560).

## Running MemGPT locally

Install MemGPT:

```sh
pip install pymemgpt
```

Add your OpenAI API key to your environment:

```sh

export OPENAI_API_KEY=YOUR_API_KEY # on Linux/Mac
set OPENAI_API_KEY=YOUR_API_KEY # on Windows
$Env:OPENAI_API_KEY = "YOUR_API_KEY" # on Windows (PowerShell)
```
Configure default setting for MemGPT by running:
```
memgpt configure
```
Now, you can run MemGPT with:
```sh
memgpt run
```
The `run` command supports the following optional flags (if set, will override config defaults):
* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--human`: (str) Name of the human to run the agent with.
* `--persona`: (str) Name of agent persona to use.
* `--model`: (str) LLM model to run [gpt-4, gpt-3.5].
* `--preset`: (str) MemGPT preset to run agent with.
* `--data_source`: (str) Name of data source (loaded with `memgpt load`) to connect to agent.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no_verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

You can run the following commands in the MemGPT CLI prompt:
* `/exit`: Exit the CLI
* `/save`: Save a checkpoint of the current agent/conversation state
* `/dump`: View the current message log (see the contents of main context)
* `/memory`: Print the current contents of agent memory
* `/pop`: Undo the last message in the conversation
* `/heartbeat`: Send a heartbeat system message to the agent
* `/memorywarning`: Send a memory warning system message to the agent


Once you exit the CLI with `/exit`, you can resume chatting with the same agent by specifying the agent name in `memgpt run --agent <NAME>`.

### Adding Custom Personas/Humans
You can add new human or persona definitions either by providing a file (using the `-f` flag) or text (using the `--text` flag).
```
# add a human
memgpt add human [-f <FILENAME>] [--text <TEXT>]

# add a persona
memgpt add persona [-f <FILENAME>] [--text <TEXT>]
```

You can view available persona and human files with the following command:
```
memgpt list [human/persona]
```

### Data Sources (i.e. chat with your data)
MemGPT supports pre-loading data into archival memory, so your agent can reference loaded data in your conversations with an agent by specifying the data source with the flag `memgpt run --data-source <NAME>`.

#### Loading Data
We currently support loading from a directory and database dumps. We highly encourage contributions for new data sources, which can be added as a new [CLI data load command](https://github.com/cpacker/MemGPT/blob/main/memgpt/cli/cli_load.py).

Loading from a directorsy:
```
# loading a directory
memgpt load directory --name <NAME> \
    [--input_dir <DIRECTORY>] [--input-files <FILE1> <FILE2>...] [--recursive]
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
To encourage your agent to reference its archival memory, we recommend adding phrases like "search your archival memory..." for the best results.

#### Viewing available data sources
You can view loaded data source with:
```
memgpt list sources
```

### Using other endpoints

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





<details>
<summary><h2>Deprecated API</h2></summary>
<details>
<summary><strong>Debugging command not found</strong></summary>

If you get `command not found` (Linux/MacOS), or a `CommandNotFoundException` (Windows), the directory where pip installs scripts is not in your PATH. You can either add that directory to your path (`pip show pip | grep Scripts`) or instead just run:
```sh
python -m memgpt
```
</details>

<details>
<summary><strong>Building from source</strong></summary>

Clone this repo: `git clone https://github.com/cpacker/MemGPT.git`

Using poetry:
1. Install poetry: `pip install poetry`
2. Run `poetry install`
3. Run `poetry run memgpt`

Using pip:
1. Run `pip install -e .`
2. Run `python3 main.py`
</details>


If you're using Azure OpenAI, set these variables instead:

```sh
# see https://github.com/openai/openai-python#microsoft-azure-endpoints
export AZURE_OPENAI_KEY = ...
export AZURE_OPENAI_ENDPOINT = ...
export AZURE_OPENAI_VERSION = ...

# set the below if you are using deployment ids
export AZURE_OPENAI_DEPLOYMENT = ...
export AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = ...

# then use the --use_azure_openai flag
memgpt --use_azure_openai
```

To create a new starter user or starter persona (that MemGPT gets initialized with), create a new `.txt` file in `~/.memgpt/humans` or `~/.memgpt/personas`, then use the `--persona` or `--human` flag when running `main.py`. For example:
```sh
# assuming you created a new file ~/.memgpt/humans/me.txt
memgpt
# Select me.txt during configuration process
```
-- OR --
```sh
# assuming you created a new file ~/.memgpt/humans/me.txt
memgpt --human me.txt
```
You can also specify any of the starter users in [/memgpt/humans/examples](/memgpt/humans/examples) or any of the starter personas in [/memgpt/personas/examples](/memgpt/personas/examples).

### GPT-3.5 support
You can run MemGPT with GPT-3.5 as the LLM instead of GPT-4:
```sh
memgpt
# Select gpt-3.5 during configuration process
```
-- OR --
```sh
memgpt --model gpt-3.5-turbo
```

**Note that this is experimental gpt-3.5-turbo support. It's quite buggy compared to gpt-4, but it should be runnable.**

Please report any bugs you encounter regarding MemGPT running on GPT-3.5 to  https://github.com/cpacker/MemGPT/issues/59.

### Local LLM support
You can run MemGPT with local LLMs too. See [instructions here](/memgpt/local_llm) and report any bugs/improvements here https://github.com/cpacker/MemGPT/discussions/67.

### `main.py` flags
```text
--first
  allows you to send the first message in the chat (by default, MemGPT will send the first message)
--debug
  enables debugging output
```

<details>
<summary>Configure via legacy flags</summary>

```text
--model
  select which model to use ('gpt-4', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo')
--persona
  load a specific persona file
--human
  load a specific human file
--archival_storage_faiss_path=<ARCHIVAL_STORAGE_FAISS_PATH>
  load in document database (backed by FAISS index)
--archival_storage_files="<ARCHIVAL_STORAGE_FILES_GLOB_PATTERN>"
  pre-load files into archival memory
--archival_storage_files_compute_embeddings="<ARCHIVAL_STORAGE_FILES_GLOB_PATTERN>"
  pre-load files into archival memory and also compute embeddings for embedding search
--archival_storage_sqldb=<SQLDB_PATH>
  load in SQL database
```
</details>


### Interactive CLI commands

These are the commands for the CLI, **not the Discord bot**! The Discord bot has separate commands you can see in Discord by typing `/`.

While using MemGPT via the CLI (not Discord!) you can run various commands:

```text
//
  toggle multiline input mode
/exit
  exit the CLI
/save
  save a checkpoint of the current agent/conversation state
/load
  load a saved checkpoint
/dump
  view the current message log (see the contents of main context)
/memory
  print the current contents of agent memory
/pop
  undo the last message in the conversation
/heartbeat
  send a heartbeat system message to the agent
/memorywarning
  send a memory warning system message to the agent
```
## Example applications
<details open>
<summary><h3>Use MemGPT to talk to your Database!</h3></summary>

MemGPT's archival memory let's you load your database and talk to it! To motivate this use-case, we have included a toy example.

Consider the `test.db` already included in the repository.

id	| name |	age
--- | --- | ---
1	| Alice |	30
2	| Bob	 | 25
3	| Charlie |	35

To talk to this database, run:

```sh
memgpt --archival_storage_sqldb=memgpt/personas/examples/sqldb/test.db
```

And then you can input the path to your database, and your query.

```python
Please enter the path to the database. test.db
...
Enter your message: How old is Bob?
...
🤖 Bob is 25 years old.
```
</details>
<details>
 <summary><h3>Loading local files into archival memory</h3></summary>
 MemGPT enables you to chat with your data locally -- this example gives the workflow for loading documents into MemGPT's archival memory.

To run our example where you can search over the SEC 10-K filings of Uber, Lyft, and Airbnb,

1. Download the .txt files from [Hugging Face](https://huggingface.co/datasets/MemGPT/example-sec-filings/tree/main) and place them in `memgpt/personas/examples/preload_archival`.

2. In the root `MemGPT` directory, run
    ```bash
    memgpt --archival_storage_files="memgpt/personas/examples/preload_archival/*.txt" --persona=memgpt_doc --human=basic
    ```

If you would like to load your own local files into MemGPT's archival memory, run the command above but replace `--archival_storage_files="memgpt/personas/examples/preload_archival/*.txt"` with your own file glob expression (enclosed in quotes).

#### Enhance with embeddings search
In the root `MemGPT` directory, run
  ```bash
  memgpt main.py --archival_storage_files_compute_embeddings="<GLOB_PATTERN>" --persona=memgpt_doc --human=basic
  ```

This will generate embeddings, stick them into a FAISS index, and write the index to a directory, and then output:
```
  To avoid computing embeddings next time, replace --archival_storage_files_compute_embeddings=<GLOB_PATTERN> with
    --archival_storage_faiss_path=<DIRECTORY_WITH_EMBEDDINGS> (if your files haven't changed).
```

If you want to reuse these embeddings, run
```bash
memgpt --archival_storage_faiss_path="<DIRECTORY_WITH_EMBEDDINGS>" --persona=memgpt_doc --human=basic
```


</details>
<details>
<summary><h3>Talking to LlamaIndex API Docs</h3></summary>

MemGPT also enables you to chat with docs -- try running this example to talk to the LlamaIndex API docs!

1.
    a. Download LlamaIndex API docs and FAISS index from [Hugging Face](https://huggingface.co/datasets/MemGPT/llamaindex-api-docs).
   ```bash
   # Make sure you have git-lfs installed (https://git-lfs.com)
   git lfs install
   git clone https://huggingface.co/datasets/MemGPT/llamaindex-api-docs
   mv llamaindex-api-docs
   ```

    **-- OR --**

   b. Build the index:
    1. Build `llama_index` API docs with `make text`. Instructions [here](https://github.com/run-llama/llama_index/blob/main/docs/DOCS_README.md). Copy over the generated `_build/text` folder to `memgpt/personas/docqa`.
    2. Generate embeddings and FAISS index.
        ```bash
        cd memgpt/personas/docqa
        python3 scrape_docs.py
        python3 generate_embeddings_for_docs.py all_docs.jsonl
        python3 build_index.py --embedding_files all_docs.embeddings.jsonl --output_index_file all_docs.index

3. In the root `MemGPT` directory, run
    ```bash
    memgpt --archival_storage_faiss_path=<ARCHIVAL_STORAGE_FAISS_PATH> --persona=memgpt_doc --human=basic
    ```
    where `ARCHIVAL_STORAGE_FAISS_PATH` is the directory where `all_docs.jsonl` and `all_docs.index` are located.
   If you downloaded from Hugging Face, it will be `memgpt/personas/docqa/llamaindex-api-docs`.
   If you built the index yourself, it will be `memgpt/personas/docqa`.
</details>
</details>

## Support

If you have any further questions, or have anything to share, we are excited to hear your feedback!

* By default MemGPT will use `gpt-4`, so your API key will require `gpt-4` API access
* For issues and feature requests, please [open a GitHub issue](https://github.com/cpacker/MemGPT/issues) or message us on our `#support` channel on [Discord](https://discord.gg/9GEQrxmVyE)

## Datasets
Datasets used in our [paper](https://arxiv.org/abs/2310.08560) can be downloaded at [Hugging Face](https://huggingface.co/MemGPT).

## 🚀 Project Roadmap
- [x] Release MemGPT Discord bot demo (perpetual chatbot)
- [x] Add additional workflows (load SQL/text into MemGPT external context)
- [x] Integration tests
- [x] Integrate with AutoGen ([discussion](https://github.com/cpacker/MemGPT/discussions/65))
- [x] Add official gpt-3.5-turbo support ([discussion](https://github.com/cpacker/MemGPT/discussions/66))
- [x] CLI UI improvements ([issue](https://github.com/cpacker/MemGPT/issues/11))
- [x] Add support for other LLM backends ([issue](https://github.com/cpacker/MemGPT/issues/18), [discussion](https://github.com/cpacker/MemGPT/discussions/67))
- [ ] Release MemGPT family of open models (eg finetuned Mistral) ([discussion](https://github.com/cpacker/MemGPT/discussions/67))

## Development

_Reminder: if you do not plan on modifying the source code, simply install MemGPT with `pip install pymemgpt`!_

First, install Poetry using [the official instructions here](https://python-poetry.org/docs/#installing-with-the-official-installer).

Then, you can install MemGPT from source with:
```
git clone git@github.com:cpacker/MemGPT.git
poetry shell
poetry install
```
We recommend installing pre-commit to ensure proper formatting during development:
```
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Contributing
We welcome pull requests! Please run the formatter before submitting a pull request:
```
poetry run black . -l 140
```
