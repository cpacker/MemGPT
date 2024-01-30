<p align="center">
  <a href="https://memgpt.ai"><img src="https://github.com/cpacker/MemGPT/assets/5475622/80f2f418-ef92-4f7a-acab-5d642faa4991" alt="MemGPT logo"></a>
</p>

<div align="center">

 <strong>MemGPT allows you to build LLM agents with self-editing memory</strong>

 <strong>Try out our MemGPT chatbot on <a href="https://discord.gg/9GEQrxmVyE">Discord</a>!</strong>
 
 <strong>You can now run MemGPT with <a href="https://memgpt.readme.io/docs/local_llm">open/local LLMs</a> and <a href="https://memgpt.readme.io/docs/autogen">AutoGen</a>!</strong>


[![Discord](https://img.shields.io/discord/1161736243340640419?label=Discord&logo=discord&logoColor=5865F2&style=flat-square&color=5865F2)](https://discord.gg/9GEQrxmVyE)
[![arxiv 2310.08560](https://img.shields.io/badge/arXiv-2310.08560-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2310.08560)
[![Documentation](https://img.shields.io/github/v/release/cpacker/MemGPT?label=Documentation&logo=readthedocs&style=flat-square)](https://memgpt.readme.io/docs)

</div>

<details open>
  <summary><h2>🤖 Create perpetual chatbots with self-editing memory!</h2></summary>
  <div align="center">
    <br>
    <img src="https://research.memgpt.ai/assets/img/demo.gif" alt="MemGPT demo video" width="800">
  </div>
</details>

<details>
 <summary><h2>🗃️ Chat with your data - talk to your local files or SQL database!</strong></h2></summary>
  <div align="center">
    <img src="https://research.memgpt.ai/assets/img/doc.gif" alt="MemGPT demo video for sql search" width="800">
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
 <img src="https://research.memgpt.ai/assets/img/discord/dm_settings.png" alt="set DMs settings on MemGPT server to be open in MemGPT so that MemGPT Bot can message you" width="400">
</div>

You can see the full list of available commands when you enter `/` into the message box.
<div align="center">
 <img src="https://research.memgpt.ai/assets/img/discord/slash_commands.png" alt="MemGPT Bot slash commands" width="400">
</div>

## What is MemGPT?
Memory-GPT (or MemGPT in short) is a system that intelligently manages different memory tiers in LLMs in order to effectively provide extended context within the LLM's limited context window. For example, MemGPT knows when to push critical information to a vector database and when to retrieve it later in the chat, enabling perpetual conversations. Learn more about MemGPT in our [paper](https://arxiv.org/abs/2310.08560).

## Running MemGPT locally
Install MemGPT:
```sh
pip install -U pymemgpt
```

Now, you can run MemGPT and start chatting with a MemGPT agent with:
```sh
memgpt run
```

If you're running MemGPT for the first time, you'll see two quickstart options:

1. **OpenAI**: select this if you'd like to run MemGPT with OpenAI models like GPT-4 (requires an OpenAI API key)
2. **MemGPT Free Endpoint**: select this if you'd like to try MemGPT on a top open LLM for free (currently variants of Mixtral 8x7b!)

Neither of these options require you to have an LLM running on your own machine. If you'd like to run MemGPT with your custom LLM setup (or on OpenAI Azure), select **Other** to proceed to the advanced setup.

### Advanced setup
You can reconfigure MemGPT's default settings by running:
```sh
memgpt configure
```

### In-chat commands
You can run the following commands in the MemGPT CLI prompt while chatting with an agent:
* `/exit`: Exit the CLI
* `/attach`: Attach a loaded data source to the agent
* `/save`: Save a checkpoint of the current agent/conversation state
* `/dump`: View the current message log (see the contents of main context)
* `/dump <count>`: View the last <count> messages (all if <count> is omitted)
* `/memory`: Print the current contents of agent memory
* `/pop`: Undo the last message in the conversation
* `/pop <count>`: Undo the last messages in the conversation. It defaults to 3, which usually is one turn around in the conversation
* `/retry`: Pops the last answer and tries to get another one
* `/rethink <text>`: Will replace the inner dialog of the last assistant message with the `<text>` to help shaping the conversation
* `/rewrite`: Will replace the last assistant answer with the given text to correct or force the answer
* `/heartbeat`: Send a heartbeat system message to the agent
* `/memorywarning`: Send a memory warning system message to the agent

Once you exit the CLI with `/exit`, you can resume chatting with the same agent by specifying the agent name in `memgpt run --agent <NAME>`.

## Documentation
See full documentation at: https://memgpt.readme.io

## Installing from source
To install MemGPT from source, start by cloning the repo:
```sh
git clone git@github.com:cpacker/MemGPT.git
```

Then navigate to the main `MemGPT` directory, and do:
```sh
pip install -e .
```

Now, you should be able to run `memgpt` from the command-line using the downloaded source code.

If you are having dependency issues using `pip install -e .`, we recommend you install the package using Poetry (see below). Installing MemGPT from source using Poetry will ensure that you are using exact package versions that have been tested for the production build.

<details>
 <summary>
  <strong>Installing from source (using Poetry)</strong>
 </summary>

First, install Poetry using [the official instructions here](https://python-poetry.org/docs/#installing-with-the-official-installer).

Then, you can install MemGPT from source with:
```sh
git clone git@github.com:cpacker/MemGPT.git
poetry shell
poetry install
```
</details>

## Python integration (for developers)

The fastest way to integrate MemGPT with your own Python projects is through the `MemGPT` client class:
```python
from memgpt import MemGPT

# Create a MemGPT client object (sets up the persistent state)
client = MemGPT(
  quickstart="openai",
  config={
    "openai_api_key": "YOUR_API_KEY"
  }
)

# You can set many more parameters, this is just a basic example
agent_id = client.create_agent(
  agent_config={
    "persona": "sam_pov",
    "human": "cs_phd",
  }
)

# Now that we have an agent_name identifier, we can send it a message!
# The response will have data from the MemGPT agent
my_message = "Hi MemGPT! How's it going?"
response = client.user_message(agent_id=agent_id, message=my_message)
```

<details>
 <summary>
  <strong>More in-depth example of using MemGPT Client</strong>
 </summary>

```python
from memgpt.config import AgentConfig
from memgpt import MemGPT
from memgpt.cli.cli import QuickstartChoice


client = MemGPT(
    # When auto_save is 'True' then the agent(s) will be saved after every
    # user message.  This may have performance implications, so you
    # can otherwise choose when to save explicitly using client.save().
    auto_save=True,

    # Quickstart will automatically configure MemGPT (without having to run `memgpt configure`
    # If you choose 'openai' then you must set the api key (env or in config)
    quickstart=QuickstartChoice.memgpt_hosted,

    # Allows you to override default config generated by quickstart or `memgpt configure`
    config={}
)

# Create an AgentConfig with default persona and human txt
# In this case, assume we wrote a custom persona file "my_persona.txt", located at ~/.memgpt/personas/my_persona.txt
# Same for a custom user file "my_user.txt", located at ~/.memgpt/humans/my_user.txt
agent_config = AgentConfig(
    name="CustomAgent",
    persona="my_persona",
    human="my_user",
)

# Create the agent according to AgentConfig we set up. If an agent with
# the same name already exists it will simply return, unless you set
# throw_if_exists to 'True'
agent_id = client.create_agent(agent_config=agent_config)

# Create a helper that sends a message and prints the assistant response only
def send_message(message: str):
    """
    sends a message and prints the assistant output only.
    :param message: the message to send
    """
    response = client.user_message(agent_id=agent_id, message=message)
    for r in response:
        # Can also handle other types "function_call", "function_return", "function_message"
        if "assistant_message" in r:
            print("ASSISTANT:", r["assistant_message"])
        elif "thoughts" in r:
            print("THOUGHTS:", r["internal_monologue"])

# Send a message and see the response
send_message("Please introduce yourself and tell me about your abilities!")
```

</details>

## What open LLMs work well with MemGPT?
When using MemGPT with open LLMs (such as those downloaded from HuggingFace), the performance of MemGPT will be highly dependent on the LLM's function calling ability.

You can find a list of LLMs/models that are known to work well with MemGPT on the [#model-chat channel on Discord](https://discord.gg/9GEQrxmVyE), as well as on [this spreadsheet](https://docs.google.com/spreadsheets/d/1fH-FdaO8BltTMa4kXiNCxmBCQ46PRBVp3Vn6WbPgsFs/edit?usp=sharing).

### Benchmarking an LLM on MemGPT (`memgpt benchmark` command)
To evaluate the performance of a model on MemGPT, simply configure the appropriate model settings using `memgpt configure`, and then initiate the benchmark via `memgpt benchmark`. The duration will vary depending on your hardware. This will run through a predefined set of prompts through multiple iterations to test the function calling capabilities of a model.

You can help track what LLMs work well with MemGPT by contributing your benchmark results via [this form](https://forms.gle/XiBGKEEPFFLNSR348), which will be used to update the spreadsheet.

## Support
For issues and feature requests, please [open a GitHub issue](https://github.com/cpacker/MemGPT/issues) or message us on our `#support` channel on [Discord](https://discord.gg/9GEQrxmVyE).

## Datasets
Datasets used in our [paper](https://arxiv.org/abs/2310.08560) can be downloaded at [Hugging Face](https://huggingface.co/MemGPT).
