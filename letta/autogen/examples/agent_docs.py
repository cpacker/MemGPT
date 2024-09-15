"""Example of how to add Letta into an AutoGen groupchat and chat with docs.

See https://letta.readme.io/docs/autogen#part-4-attaching-documents-to-letta-autogen-agents

Based on the official AutoGen example here: https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat.ipynb

Begin by doing:
  pip install "pyautogen[teachable]"
  pip install pyletta
  or
  pip install -e . (inside the Letta home directory)
"""

import os

import autogen

from letta.autogen.letta_agent import create_letta_autogen_agent_from_config
from letta.constants import DEFAULT_PRESET, LLM_MAX_TOKENS

LLM_BACKEND = "openai"
# LLM_BACKEND = "azure"
# LLM_BACKEND = "local"

if LLM_BACKEND == "openai":
    # For demo purposes let's use gpt-4
    model = "gpt-4"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert openai_api_key, "You must set OPENAI_API_KEY or set LLM_BACKEND to 'local' to run this example"

    # This config is for AutoGen agents that are not powered by Letta
    config_list = [
        {
            "model": model,
            "api_key": os.getenv("OPENAI_API_KEY"),
        }
    ]

    # This config is for AutoGen agents that powered by Letta
    config_list_letta = [
        {
            "model": model,
            "context_window": LLM_MAX_TOKENS[model],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # OpenAI specific
            "model_endpoint_type": "openai",
            "model_endpoint": "https://api.openai.com/v1",
            "openai_key": openai_api_key,
        },
    ]

elif LLM_BACKEND == "azure":
    # Make sure that you have access to this deployment/model on your Azure account!
    # If you don't have access to the model, the code will fail
    model = "gpt-4"

    azure_openai_api_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    assert (
        azure_openai_api_key is not None and azure_openai_version is not None and azure_openai_endpoint is not None
    ), "Set all the required OpenAI Azure variables (see: https://letta.readme.io/docs/endpoints#azure-openai)"

    # This config is for AutoGen agents that are not powered by Letta
    config_list = [
        {
            "model": model,
            "api_type": "azure",
            "api_key": azure_openai_api_key,
            "api_version": azure_openai_version,
            # NOTE: on versions of pyautogen < 0.2.0, use "api_base"
            # "api_base": azure_openai_endpoint,
            "base_url": azure_openai_endpoint,
        }
    ]

    # This config is for AutoGen agents that powered by Letta
    config_list_letta = [
        {
            "model": model,
            "context_window": LLM_MAX_TOKENS[model],
            "preset": DEFAULT_PRESET,
            "model_wrapper": None,
            # Azure specific
            "model_endpoint_type": "azure",
            "azure_key": azure_openai_api_key,
            "azure_endpoint": azure_openai_endpoint,
            "azure_version": azure_openai_version,
        },
    ]

elif LLM_BACKEND == "local":
    # Example using LM Studio on a local machine
    # You will have to change the parameters based on your setup

    # Non-Letta agents will still use local LLMs, but they will use the ChatCompletions endpoint
    config_list = [
        {
            "model": "NULL",  # not needed
            # NOTE: on versions of pyautogen < 0.2.0 use "api_base", and also uncomment "api_type"
            # "api_base": "http://localhost:1234/v1",
            # "api_type": "open_ai",
            "base_url": "http://localhost:1234/v1",  # ex. "http://127.0.0.1:5001/v1" if you are using webui, "http://localhost:1234/v1/" if you are using LM Studio
            "api_key": "NULL",  #  not needed
        },
    ]

    # Letta-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
    config_list_letta = [
        {
            "preset": DEFAULT_PRESET,
            "model": None,  # only required for Ollama, see: https://letta.readme.io/docs/ollama
            "context_window": 8192,  # the context window of your model (for Mistral 7B-based models, it's likely 8192)
            "model_wrapper": "chatml",  # chatml is the default wrapper
            "model_endpoint_type": "lmstudio",  # can use webui, ollama, llamacpp, etc.
            "model_endpoint": "http://localhost:1234",  # the IP address of your LLM backend
        },
    ]

else:
    raise ValueError(LLM_BACKEND)

# Set to True if you want to print Letta's inner workings.
DEBUG = False

interface_kwargs = {
    "debug": DEBUG,
    "show_inner_thoughts": True,
    "show_function_outputs": True,  # let's set this to True so that we can see the search function in action
}

llm_config = {"config_list": config_list, "seed": 42}
llm_config_letta = {"config_list": config_list_letta, "seed": 42}

# The user agent
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE",  # needed?
    default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
)

# In our example, we swap this AutoGen agent with a Letta agent
# This Letta agent will have all the benefits of Letta, ie persistent memory, etc.
letta_agent = create_letta_autogen_agent_from_config(
    "Letta_agent",
    llm_config=llm_config_letta,
    system_message=f"You are an AI research assistant.\n" f"You are participating in a group chat with a user ({user_proxy.name}).",
    interface_kwargs=interface_kwargs,
    default_auto_reply="...",  # Set a default auto-reply message here (non-empty auto-reply is required for LM Studio)
    skip_verify=False,  # NOTE: you should set this to True if you expect your Letta AutoGen agent to call a function other than send_message on the first turn
)
# NOTE: you need to follow steps to load document first: see https://letta.readme.io/docs/autogen#part-4-attaching-documents-to-letta-autogen-agents
letta_agent.load_and_attach(
    name="letta_research_paper",
    type="directory",
    input_dir=None,
    input_files=["letta_research_paper.pdf"],
    # force=True,
)

# Initialize the group chat between the agents
groupchat = autogen.GroupChat(agents=[user_proxy, letta_agent], messages=[], max_round=3, speaker_selection_method="round_robin")
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Begin the group chat with a message from the user
user_proxy.initiate_chat(
    manager,
    message="Tell me what virtual context in Letta is. Search your archival memory.",
)
