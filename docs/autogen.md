## MemGPT + Autogen

!!! warning "Need help?"

    If you need help visit our [Discord server](https://discord.gg/9GEQrxmVyE) and post in the #support channel.
    
    You can also check the [GitHub discussion page](https://github.com/cpacker/MemGPT/discussions/65), but the Discord server is the official support channel and is monitored more actively.

[examples/agent_groupchat.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_groupchat.py) contains an example of a groupchat where one of the agents is powered by MemGPT.

If you are using OpenAI, you can also run it using the [example notebook](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/memgpt_coder_autogen.ipynb).

In the next section, we detail how to set up MemGPT and Autogen to run with local LLMs.

## Example: connecting Autogen + MemGPT to non-OpenAI LLMs (using oobabooga web UI)

!!! warning "Enable the OpenAI extension"

    In web UI make sure to enable the [openai extension](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)!
    
    This is enabled by default in newer versions of web UI, but must be enabled manually in older versions of web UI.

To get MemGPT to work with a local LLM, you need to have an LLM running on a server that takes API requests.

For the purposes of this example, we're going to serve (host) the LLMs using [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui), but if you want to use something else you can! This also assumes your running web UI locally - if you're running on e.g. Runpod, you'll want to follow Runpod specific instructions (for example use [TheBloke's one-click UI and API](https://github.com/TheBlokeAI/dockerLLM/blob/main/README_Runpod_LocalLLMsUIandAPI.md)).

### Part 1: Get web UI working

Install web UI and get a model set up on a local web server. You can use [our instructions on setting up web UI](https://memgpt.readthedocs.io/en/latest/webui/).

!!! warning "Choosing an LLM / model to use"

    You'll need to decide on an LLM / model to use with web UI.
    
    MemGPT requires an LLM that is good at function calling to work well - if the LLM is bad at function calling, **MemGPT will not work properly**.

    Visit [our Discord server](https://discord.gg/9GEQrxmVyE) for an up-to-date list of recommended LLMs / models to use with MemGPT.

### Part 2: Get MemGPT working

Before trying to integrate MemGPT with AutoGen, make sure that you can run MemGPT by itself with the web UI backend.

Try setting up MemGPT with your local web UI backend [using the instructions here](https://memgpt.readthedocs.io/en/latest/local_llm/#using-memgpt-with-local-llms).

Once you've confirmed that you're able to chat with a MemGPT agent using `memgpt configure` and `memgpt run`, you're ready to move on to the next step.

!!! warning "Using RunPod as an LLM backend"

    If you're using RunPod to run web UI, make sure that you set your endpoint to the RunPod IP address, **not the default localhost address**.

    For example, during `memgpt configure`: `? Enter default endpoint: https://yourpodaddresshere-5000.proxy.runpod.net`

### Part 3: Creating a MemGPT AutoGen agent (groupchat example)

Now we're going to integrate MemGPT and AutoGen by creating a special "MemGPT AutoGen agent" that wraps MemGPT in an AutoGen-style agent interface.

First, make sure you have AutoGen installed (choose only ONE of the following methods depending on how you installed MemGPT):
```sh
# if you installed MemGPT with `pip install pymemgpt`
pip install pymemgpt[autogen]

# if you installed MemGPT with `git clone`, `cd MemGPT`, `pip install -e .`
pip install -e .[autogen]

# if you installed MemGPT with `poetry install`
poetry install -E autogen
```

Going back to the example we first mentioned, [examples/agent_groupchat.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_groupchat.py) contains an example of a groupchat where one of the agents is powered by MemGPT.

In order to run this example on a local LLM, go to lines 46-66 in [examples/agent_groupchat.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_groupchat.py) and fill in the config files with your local LLM's deployment details.

`config_list` is used by non-MemGPT AutoGen agents, which expect an OpenAI-compatible API. `config_list_memgpt` is used by MemGPT AutoGen agents, and requires additional settings specific to MemGPT (such as the `model_wrapper` and `context_window`.

For example, if you are using web UI, it will look something like this:
```python
    # Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
    config_list = [
        {
            "model": "NULL",  # not needed
            "api_base": "http://127.0.0.1:5001/v1",  # notice port 5001 for web UI
            "api_key": "NULL",  #  not needed
            "api_type": "open_ai",
        },
    ]

    # MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
    config_list_memgpt = [
        {
            "preset": DEFAULT_PRESET,
            "model": None,  # not required for web UI, only required for Ollama, see: https://memgpt.readthedocs.io/en/latest/ollama/
            "model_wrapper": "airoboros-l2-70b-2.1",  # airoboros is the default wrapper and should work for most models
            "model_endpoint_type": "webui",
            "model_endpoint": "http://localhost:5000",  # notice port 5000 for web UI
            "context_window": 8192,  # the context window of your model (for Mistral 7B-based models, it's likely 8192)
        },
    ]
```

If you are using LM Studio, then you'll need to change the `api_base` in `config_list`, and `model_endpoint_type` + `model_endpoint` in `config_list_memgpt`:
```python
    # Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
    config_list = [
        {
            "model": "NULL",
            "api_base": "http://127.0.0.1:1234/v1",  # port 1234 for LM Studio
            "api_key": "NULL",
            "api_type": "open_ai",
        },
    ]

    # MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
    config_list_memgpt = [
        {
            "preset": DEFAULT_PRESET,
            "model": None,
            "model_wrapper": "airoboros-l2-70b-2.1",
            "model_endpoint_type": "lmstudio",
            "model_endpoint": "http://localhost:1234",  # port 1234 for LM Studio
            "context_window": 8192,
        },
    ]
```

If you are using the OpenAI API (e.g. using `gpt-4-turbo` via your own OpenAI API account), then the `config_list` for the AutoGen agent and `config_list_memgpt` for the MemGPT AutoGen agent will look different (a lot simpler):
```python
    # This config is for autogen agents that are not powered by MemGPT
    config_list = [
        {
            "model": "gpt-4-1106-preview",  # gpt-4-turbo (https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ]

    # This config is for autogen agents that powered by MemGPT
    config_list_memgpt = [
        {
            "model": "gpt-4-1106-preview",  # gpt-4-turbo (https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
            "preset": DEFAULT_PRESET,
            "model": None,
            "model_wrapper": None,
            "model_endpoint_type": None,
            "model_endpoint": None,
            "context_window": 128000,  # gpt-4-turbo
        },
    ]
```

The only parts of the `agent_groupchat.py` file you need to modify should be the `config_list` and `config_list_memgpt` (make sure to change `USE_OPENAI` to `True` or `False` depending on if you're trying to use a local LLM server like web UI, or OpenAI's API). Assuming you edited things correctly, you should now be able to run `agent_groupchat.py`:
```sh
python memgpt/autogen/examples/agent_groupchat.py
```

## Loading documents

[examples/agent_docs.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_docs.py) contains an example of a groupchat where the MemGPT autogen agent has access to documents.

First, follow the instructions in [Example - chat with your data - Creating an external data source](../example_data/#creating-an-external-data-source):

To download the MemGPT research paper we'll use `curl` (you can also just download the PDF from your browser):
```sh
# we're saving the file as "memgpt_research_paper.pdf"
curl -L -o memgpt_research_paper.pdf https://arxiv.org/pdf/2310.08560.pdf
```

Now that we have the paper downloaded, we can create a MemGPT data source using `memgpt load`:
```sh
memgpt load directory --name memgpt_research_paper --input-files=memgpt_research_paper.pdf
```
```text
loading data
done loading data
LLM is explicitly disabled. Using MockLLM.
LLM is explicitly disabled. Using MockLLM.
Parsing documents into nodes: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 392.09it/s]
Generating embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:01<00:00, 37.34it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:00<00:00, 388361.48it/s]
Saved local /home/user/.memgpt/archival/memgpt_research_paper/nodes.pkl
```

Note: you can ignore the "_LLM is explicitly disabled_" message.

Now, you can run `agent_docs.py`, which asks `MemGPT_coder` what a virtual context is:
```
❯ python3 agent_docs.py
LLM is explicitly disabled. Using MockLLM.
LLM is explicitly disabled. Using MockLLM.
LLM is explicitly disabled. Using MockLLM.
Generating embeddings: 0it [00:00, ?it/s]
new size 60
Saved local /Users/vivian/.memgpt/agents/agent_25/persistence_manager/index/nodes.pkl
Attached data source memgpt_research_paper to agent agent_25, consisting of 60. Agent now has 60 embeddings in archival memory.
LLM is explicitly disabled. Using MockLLM.
User_proxy (to chat_manager):

Tell me what a virtual context in MemGPT is. Search your archival memory.

--------------------------------------------------------------------------------
GroupChat is underpopulated with 2 agents. Direct communication would be more efficient.

MemGPT_coder (to chat_manager):

Virtual context management is a technique used in large language models like MemGPT. It's used to handle context beyond limited context windows, which is crucial for tasks such as extended conversations and document analysis. The technique was inspired by hierarchical memory systems in traditional operating systems that provide the appearance of large memory resources through data movement between fast and slow memory. This system intelligently manages different memory tiers to effectively provide extended context within the model's limited context window.

--------------------------------------------------------------------------------
...
```
