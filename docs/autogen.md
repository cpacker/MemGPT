---
title: MemGPT + AutoGen
excerpt: Creating AutoGen agents powered by MemGPT
category: 6580dab16cade8003f996d17
---

> 📘 Need help?
>
> If you need help visit our [Discord server](https://discord.gg/9GEQrxmVyE) and post in the #support channel.
>
> You can also check the [GitHub discussion page](https://github.com/cpacker/MemGPT/discussions/65), but the Discord server is the official support channel and is monitored more actively.

> ⚠️ Tested with `pyautogen` v0.2.0
>
> The MemGPT+AutoGen integration was last tested using AutoGen version v0.2.0.
>
> If you are having issues, please first try installing the specific version of AutoGen using `pip install pyautogen==0.2.0` (or `poetry install -E autogen` if you are using Poetry).

## Overview

MemGPT includes an AutoGen agent class ([MemGPTAgent](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/memgpt_agent.py)) that mimics the interface of AutoGen's [ConversableAgent](https://microsoft.github.io/autogen/docs/reference/agentchat/conversable_agent#conversableagent-objects), allowing you to plug MemGPT into the AutoGen framework.

To create a MemGPT AutoGen agent for use in an AutoGen script, you can use the `create_memgpt_autogen_agent_from_config` constructor:

```python
from memgpt.autogen.memgpt_agent import create_memgpt_autogen_agent_from_config

# create a config for the MemGPT AutoGen agent
config_list_memgpt = [
    {
        "model": "gpt-4",
        "context_window": 8192,
        "preset": "memgpt_chat",  # NOTE: you can change the preset here
        # OpenAI specific
        "model_endpoint_type": "openai",
        "openai_key": YOUR_OPENAI_KEY,
    },
]
llm_config_memgpt = {"config_list": config_list_memgpt, "seed": 42}

# there are some additional options to do with how you want the interface to look (more info below)
interface_kwargs = {
    "debug": False,
    "show_inner_thoughts": True,
    "show_function_outputs": False,
}

# then pass the config to the constructor
memgpt_autogen_agent = create_memgpt_autogen_agent_from_config(
    "MemGPT_agent",
    llm_config=llm_config_memgpt,
    system_message=f"Your desired MemGPT persona",
    interface_kwargs=interface_kwargs,
    default_auto_reply="...",
    skip_verify=False,  # NOTE: you should set this to True if you expect your MemGPT AutoGen agent to call a function other than send_message on the first turn
)
```

Now this `memgpt_autogen_agent` can be used in standard AutoGen scripts:

```python
import autogen

# ... assuming we have some other AutoGen agents other_agent_1 and 2
groupchat = autogen.GroupChat(agents=[memgpt_autogen_agent, other_agent_1, other_agent_2], messages=[], max_round=12)
```

[examples/agent_groupchat.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_groupchat.py) contains an example of a groupchat where one of the agents is powered by MemGPT. If you are using OpenAI, you can also run the example using the [notebook](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/memgpt_coder_autogen.ipynb).

In the next section, we'll go through the example in depth to demonstrate how to set up MemGPT and AutoGen to run with a local LLM backend.

## Example: connecting AutoGen + MemGPT to non-OpenAI LLMs

To get MemGPT to work with a local LLM, you need to have an LLM running on a server that takes API requests.

For the purposes of this example, we're going to serve (host) the LLMs using [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui), but if you want to use something else you can! This also assumes your running web UI locally - if you're running on e.g. Runpod, you'll want to follow Runpod specific instructions (for example use [TheBloke's one-click UI and API](https://github.com/TheBlokeAI/dockerLLM/blob/main/README_Runpod_LocalLLMsUIandAPI.md)).

### Part 1: Get web UI working

Install web UI and get a model set up on a local web server. You can use [our instructions on setting up web UI](webui).

> 📘 Choosing an LLM / model to use
> You'll need to decide on an LLM / model to use with web UI.
>
> MemGPT requires an LLM that is good at function calling to work well - if the LLM is bad at function calling, **MemGPT will not work properly**.
>
> Visit [our Discord server](https://discord.gg/9GEQrxmVyE) and check the #model-chat channel for an up-to-date list of recommended LLMs / models to use with MemGPT.

### Part 2: Get MemGPT working

Before trying to integrate MemGPT with AutoGen, make sure that you can run MemGPT by itself with the web UI backend.

Try setting up MemGPT with your local web UI backend [using the instructions here](local_llm/#using-memgpt-with-local-llms).

Once you've confirmed that you're able to chat with a MemGPT agent using `memgpt configure` and `memgpt run`, you're ready to move on to the next step.

> 📘 Using RunPod as an LLM backend
>
> If you're using RunPod to run web UI, make sure that you set your endpoint to the RunPod IP address, **not the default localhost address**.
>
> For example, during `memgpt configure`:
>
> ```text
> ? Enter default endpoint: https://yourpodaddresshere-5000.proxy.runpod.net
> ```

### Part 3: Creating a MemGPT AutoGen agent (groupchat example)

Now we're going to integrate MemGPT and AutoGen by creating a special "MemGPT AutoGen agent" that wraps MemGPT in an AutoGen-style agent interface.

First, make sure you have AutoGen installed:

```sh
pip install pyautogen
```

Going back to the example we first mentioned, [examples/agent_groupchat.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_groupchat.py) contains an example of a groupchat where one of the agents is powered by MemGPT.

In order to run this example on a local LLM, go to lines 46-66 in [examples/agent_groupchat.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_groupchat.py) and fill in the config files with your local LLM's deployment details.

`config_list` is used by non-MemGPT AutoGen agents, which expect an OpenAI-compatible API. `config_list_memgpt` is used by MemGPT AutoGen agents, and requires additional settings specific to MemGPT (such as the `model_wrapper` and `context_window`. Depending on what LLM backend you want to use, you'll have to set up your `config_list` and `config_list_memgpt` differently:

#### web UI example

For example, if you are using web UI, it will look something like this:

```python
# Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
config_list = [
    {
        "model": "NULL",  # not needed
        "base_url": "http://127.0.0.1:5001/v1",  # notice port 5001 for web UI
        "api_key": "NULL",  #  not needed
    },
]

# MemGPT-powered agents will also use local LLMs, but they need additional setup (also they use the Completions endpoint)
config_list_memgpt = [
    {
        "preset": DEFAULT_PRESET,
        "model": None,  # not required for web UI, only required for Ollama, see: https://memgpt.readme.io/docs/ollama
        "model_wrapper": "airoboros-l2-70b-2.1",  # airoboros is the default wrapper and should work for most models
        "model_endpoint_type": "webui",
        "model_endpoint": "http://localhost:5000",  # notice port 5000 for web UI
        "context_window": 8192,  # the context window of your model (for Mistral 7B-based models, it's likely 8192)
    },
]
```

#### LM Studio example

If you are using LM Studio, then you'll need to change the `api_base` in `config_list`, and `model_endpoint_type` + `model_endpoint` in `config_list_memgpt`:

```python
# Non-MemGPT agents will still use local LLMs, but they will use the ChatCompletions endpoint
config_list = [
    {
        "model": "NULL",
        "base_url": "http://127.0.0.1:1234/v1",  # port 1234 for LM Studio
        "api_key": "NULL",
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

#### OpenAI example

If you are using the OpenAI API (e.g. using `gpt-4-turbo` via your own OpenAI API account), then the `config_list` for the AutoGen agent and `config_list_memgpt` for the MemGPT AutoGen agent will look different (a lot simpler):

```python
# This config is for autogen agents that are not powered by MemGPT
config_list = [
    {
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]

# This config is for autogen agents that powered by MemGPT
config_list_memgpt = [
    {
        "preset": DEFAULT_PRESET,
        "model": "gpt-4",
        "context_window": 8192,  # gpt-4 context window
        "model_wrapper": None,
        "model_endpoint_type": "openai",
        "model_endpoint": "https://api.openai.com/v1",
        "openai_key": os.getenv("OPENAI_API_KEY"),
    },
]
```

#### Azure OpenAI example

Azure OpenAI API setup will be similar to OpenAI API, but requires additional config variables. First, make sure that you've set all the related Azure variables referenced in [our MemGPT Azure setup page](https://memgpt.readme.io/docs/endpoints#azure-openai) (`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_VERSION`, `AZURE_OPENAI_ENDPOINT`, etc). If you have all the variables set correctly, you should be able to create configs by pulling from the env variables:

```python
# This config is for autogen agents that are not powered by MemGPT
# See Auto
config_list = [
    {
        "model": "gpt-4",  # make sure you choose a model that you have access to deploy on your Azure account
        "api_type": "azure",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("AZURE_OPENAI_VERSION"),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT"),
    }
]

# This config is for autogen agents that powered by MemGPT
config_list_memgpt = [
    {
        "preset": DEFAULT_PRESET,
        "model": "gpt-4",  # make sure you choose a model that you have access to deploy on your Azure account
        "model_wrapper": None,
        "context_window": 8192,  # gpt-4 context window
        # required setup for Azure
        "model_endpoint_type": "azure",
        "azure_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_version": os.getenv("AZURE_OPENAI_VERSION"),
        # if you are using Azure for embeddings too, include the following line:
        "embedding_embedding_endpoint_type": "azure",
    },
]
```

> 📘 Making internal monologue visible to AutoGen
>
> By default, MemGPT's inner monologue and function traces are hidden from other AutoGen agents.
>
> You can modify `interface_kwargs` to change the visibility of inner monologue and function calling:
>
> ```python
> interface_kwargs = {
>     "debug": False,  # this is the equivalent of the --debug flag in the MemGPT CLI
>     "show_inner_thoughts": True,  # this controls if internal monlogue will show up in AutoGen MemGPT agent's outputs
>     "show_function_outputs": True,  # this controls if function traces will show up in AutoGen MemGPT agent's outputs
> }
> ```

The only parts of the `agent_groupchat.py` file you need to modify should be the `config_list` and `config_list_memgpt` (make sure to change `USE_OPENAI` to `True` or `False` depending on if you're trying to use a local LLM server like web UI, or OpenAI's API). Assuming you edited things correctly, you should now be able to run `agent_groupchat.py`:

```sh
python memgpt/autogen/examples/agent_groupchat.py
```

Your output should look something like this:

```text
User_proxy (to chat_manager):

I want to design an app to make me one million dollars in one month. Yes, your heard that right.

--------------------------------------------------------------------------------
Product_manager (to chat_manager):

Creating an app or software product that can generate one million dollars in one month is a highly ambitious goal. To achieve such a significant financial outcome quickly, your app idea needs to appeal to a broad audience, solve a significant problem, create immense value, and have a solid revenue model. Here are a few steps and considerations that might help guide you towards that goal:

1. **Identify a Niche Market or Trend:** Look for emerging trends or underserved niches that are gaining traction. This could involve addressing new consumer behaviors, leveraging new technologies, or entering a rapidly growing industry.

2. **Solve a Real Problem:** Focus on a problem that affects a large number of people or businesses and offer a unique, effective solution. The more painful the problem, the more willing customers will be to pay for a solution.

3. **Monetization Strategy:** Decide how you will make money from your app. Common strategies include paid downloads, in-app purchases, subscription models, advertising, or a freemium model with premium features.

4. **Viral Mechanism:** Design your app so that it encourages users to share it with others, either through inherent network effects (e.g., social media platforms) or through incentives (e.g., referral programs).

5. **Marketing Campaign:** Even the best app can't make money if people don't know about it. Plan a robust marketing campaign to launch your app, using social media, influencer partnerships, press releases, and advertising.

6. **Rapid Iteration and Scaling:** Be prepared to iterate rapidly based on user feedback and scale quickly to accommodate user growth. The faster you can improve and grow, the more likely it is you'll reach your revenue target.

7. **Partnerships and Alliances:** Partner with other companies or influencers who can market your product to their user base. This could provide a significant boost to your initial user acquisition.

8. **Compliance and Security:** Ensure that your app complies with all legal requirements and has high standards of privacy and security, especially if you are handling sensitive user data.

Here are a few app ideas that have the potential to be lucrative if well executed:

- **Health and Wellness Platform:** An app that uses AI to personalize workout and nutrition plans, with a community feature for motivation and support. Monetize through subscription and premium features.

- **FinTech Solution:** An investment or savings app that simplifies the process of cryptocurrency trading or micro-investment. Make money through transaction fees or subscription services.

- **Educational Platform:** Offer a unique learning experience with expert-created content for specific skills in high demand, such as coding, design, or digital marketing. Use a subscription model with tiered pricing.

- **AR/VR Experiences:** Develop an app that provides immersive experiences for entertainment, education, or practical purposes like interior design. Charge for the app itself or offer in-app purchases.

- **Marketplace or Gig Economy App:** Create a platform that matches freelancers or service providers with people who need their services. Revenue could come from taking a cut of the transactions.

Remember, achieving one million dollars in revenue in such a short time frame would require not only a highly appealing and innovative product but also flawless execution, significant marketing efforts, and perhaps a bit of luck. Be realistic about your goals and focus on building a sustainable business that provides real value over the long term.

--------------------------------------------------------------------------------
MemGPT_coder (to chat_manager):

Great goal! Generating a million dollars in one month with an app is ambitious, but definitely doable if you approach it the right way. Here are some tips and potential ideas that could help:

1. Identify a niche market or trend (for example, AI-powered fitness apps or FinTech solutions).
2. Solve a significant problem for many people (such as time management or financial literacy).
3. Choose an effective monetization strategy like subscriptions, in-app purchases, or advertising.
4. Make sure your app is visually appealing and easy to use to keep users engaged.

Some ideas that might work:
- AI-powered personal finance management app
- A virtual assistant app that helps people manage their daily tasks
- A social networking platform for job seekers or freelancers

Remember, success often comes from focusing on a specific problem and delivering a unique solution. Good luck!

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
User_proxy (to chat_manager):

...
```

### Part 4: Attaching documents to MemGPT AutoGen agents

[examples/agent_docs.py](https://github.com/cpacker/MemGPT/blob/main/memgpt/autogen/examples/agent_docs.py) contains an example of a groupchat where the MemGPT autogen agent has access to documents.

First, follow the instructions in [Example - chat with your data - Creating an external data source](example_data/#creating-an-external-data-source):

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
Parsing documents into nodes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [00:00<00:00, 321.56it/s]
Generating embeddings: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 65/65 [00:01<00:00, 43.22it/s]
100%|██████████████████████████████████████████████
```

Note: you can ignore the "_LLM is explicitly disabled_" message.

Now, you can run `agent_docs.py`, which asks `MemGPT_coder` what a virtual context is:

```sh
python memgpt/autogen/examples/agent_docs.py
```

```text
Ingesting 65 passages into MemGPT_agent
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.47s/it]
Attached data source memgpt_research_paper to agent MemGPT_agent, consisting of 65. Agent now has 2015 embeddings in archival memory.

User_proxy (to chat_manager):

Tell me what virtual context in MemGPT is. Search your archival memory.

--------------------------------------------------------------------------------
GroupChat is underpopulated with 2 agents. Direct communication would be more efficient.

MemGPT_agent (to chat_manager):

[inner thoughts] The user asked about virtual context in MemGPT. Let's search the archival memory with this query.
[inner thoughts] Virtual context management is a technique used in large language models like MemGPT. It's used to handle context beyond limited context windows, which is crucial for tasks such as extended conversations and document analysis. The technique was inspired by hierarchical memory systems in traditional operating systems that provide the appearance of large memory resources through data movement between fast and slow memory. This system intelligently manages different memory tiers to effectively provide extended context within the model's limited context window.

--------------------------------------------------------------------------------
...
```
