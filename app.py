import os
import autogen
import memgpt.autogen.memgpt_agent as memgpt_autogen
import memgpt.autogen.interface as autogen_interface
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithFaiss
import openai

openai.api_key = "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"
openai.api_key = "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"
openai.api_key = "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"
openai.api_key = "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"
openai.api_key = "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"

config_list = [
    {"model": "gpt-4"},
]

llm_config = {"config_list": config_list, "seed": 42}
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
)

interface = autogen_interface.AutoGenInterface()  # how MemGPT talks to AutoGen
persistence_manager = InMemoryStateManager()
persona = "I'm a 10x engineer at a FAANG tech company."
human = "I'm a team manager at a FAANG tech company."
memgpt_agent = presets.use_preset(presets.DEFAULT_PRESET, "gpt-4", persona, human, interface, persistence_manager)

# MemGPT coder
coder = memgpt_autogen.MemGPTAgent(
    name="MemGPT_coder",
    agent=memgpt_agent,
)

# non-MemGPT PM
pm = autogen.AssistantAgent(
    name="Product_manager",
    system_message="Creative in software product ideas.",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(manager, message="First send the message 'Let's go Mario!'")
