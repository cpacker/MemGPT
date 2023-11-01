# MemGPT + Autogen examples
[examples/agent_groupchat.py](examples/agent_groupchat.py) contains an example of a groupchat where one of the agents is powered by MemGPT.

**Local LLM support**
In order to run MemGPT+Autogen on a local LLM, uncomment lines 31-46 in [examples/agent_groupchat.py](exmaples/agent_groupchat.py) and fill in the config files with your local LLM's deployment details. For example, if you are using webui, it will look something like this:

```
config_list = [
    {
        "model": "dolphin-2.1-mistral-7b",
        "api_base": "http://127.0.0.1:5001/v1"
        "api_key": "NULL", # this is a placeholder
        "api_type": "open_ai",
    },
]
config_list_memgpt = [
    {
        "model": "dolphin-2.1-mistral-7b",
    },
]
```

Also be sure to export `OPENAI_API_BASE` and `BACKEND_TYPE` as described in [Local LLM support](../local_llm).