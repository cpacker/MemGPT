"""Example of how to add MemGPT into an AutoGen groupchat and chat with docs.

See https://memgpt.readthedocs.io/en/latest/autogen/#loading-documents

Based on the official AutoGen example here: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb

Begin by doing:
  pip install "pyautogen[teachable]"
  pip install pymemgpt
  or
  pip install -e . (inside the MemGPT home directory)
"""


import os
import autogen
from memgpt.autogen.memgpt_agent import create_autogen_memgpt_agent, create_memgpt_autogen_agent_from_config

# USE_OPENAI = True
USE_OPENAI = False
if USE_OPENAI:
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
            "preset": "memgpt_docs",
            "model": None,
            "model_wrapper": None,
            "model_endpoint_type": None,
            "model_endpoint": None,
            "context_window": 128000,  # gpt-4-turbo
        },
    ]

else:
    # Example using LM Studio on a local machine
    # You will have to change the parameters based on your setup

    # Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
    config_list = [
        {
            "model": "NULL",  # not needed
            "api_base": "http://localhost:1234/v1",  # ex. "http://127.0.0.1:5001/v1" if you are using webui, "http://localhost:1234/v1/" if you are using LM Studio
            "api_key": "NULL",  #  not needed
            "api_type": "open_ai",
        },
    ]

    # MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
    config_list_memgpt = [
        {
            "preset": "memgpt_docs",
            "model": None,  # only required for Ollama, see: https://memgpt.readthedocs.io/en/latest/ollama/
            "model_wrapper": "airoboros-l2-70b-2.1",  # airoboros is the default wrapper and should work for most models
            "model_endpoint_type": "lmstudio",  # can use webui, ollama, llamacpp, etc.
            "model_endpoint": "http://localhost:1234",  # the IP address of your LLM backend
            "context_window": 8192,  # the context window of your model (for Mistral 7B-based models, it's likely 8192)
        },
    ]

# Set to True if you want to print MemGPT's inner workings.
DEBUG = False

interface_kwargs = {
    "debug": DEBUG,
    "show_inner_thoughts": DEBUG,
    "show_function_outputs": DEBUG,
}

llm_config = {"config_list": config_list, "seed": 42}
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

# The user agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE",  # needed?
    default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
)

# In our example, we swap this AutoGen agent with a MemGPT agent
# This MemGPT agent will have all the benefits of MemGPT, ie persistent memory, etc.
memgpt_agent = create_memgpt_autogen_agent_from_config(
    "MemGPT_agent",
    llm_config=llm_config_memgpt,
    system_message=f"I am a 10x engineer, trained in Python. I was the first engineer at Uber "
    f"(which I make sure to tell everyone I work with).\n"
    f"You are participating in a group chat with a user ({user_proxy.name}).",
    interface_kwargs=interface_kwargs,
    default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
)
# NOTE: you need to follow steps to load document first: see https://memgpt.readthedocs.io/en/latest/autogen/#loading-documents
memgpt_agent.load_and_attach("memgpt_research_paper", "directory")

# Initialize the group chat between the agents
groupchat = autogen.GroupChat(agents=[user_proxy, memgpt_agent], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    manager,
    message="Tell me what a virtual context in MemGPT is. Search your archival memory.",
)
