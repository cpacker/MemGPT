<a href="#user-content-memgpt"><img src="https://memgpt.ai/assets/img/memgpt_logo_circle.png" alt="MemGPT logo" width="75" align="right"></a>

# [MemGPT](https://memgpt.ai)

<div align="center">

 <strong>Try out our MemGPT chatbot on <a href="https://discord.gg/9GEQrxmVyE">Discord</a>!</strong>

 <strong>⭐ NEW: You can now run MemGPT with <a href="https://memgpt.readthedocs.io/en/latest/local_llm/">local LLMs</a> and <a href="https://github.com/cpacker/MemGPT/discussions/65">AutoGen</a>! ⭐ </strong>


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

You can run the following commands in the MemGPT CLI prompt:
* `/exit`: Exit the CLI
* `/attach`: Attach a loaded data source to the agent
* `/save`: Save a checkpoint of the current agent/conversation state
* `/dump`: View the current message log (see the contents of main context)
* `/dump <count>`: View the last <count> messages (all if <count> is omitted)
* `/memory`: Print the current contents of agent memory
* `/pop`: Undo the last message in the conversation
* `/pop <count>`: Undo the last messages in the conversation. It defaults to 3, which usually is one turn around in the conversation
* `/retry`: Pops the last answer and tries to get another one
* `/rethink <text>`: Will replace the inner dialog of the last assistant message with the <text> to help shaping the conversation
* `/rewrite`: Will replace the last assistant answer with the given text to correct or force the answer
* `/heartbeat`: Send a heartbeat system message to the agent
* `/memorywarning`: Send a memory warning system message to the agent


Once you exit the CLI with `/exit`, you can resume chatting with the same agent by specifying the agent name in `memgpt run --agent <NAME>`.

## Documentation
See full documentation at: https://memgpt.readthedocs.io/

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

## Support
For issues and feature requests, please [open a GitHub issue](https://github.com/cpacker/MemGPT/issues) or message us on our `#support` channel on [Discord](https://discord.gg/9GEQrxmVyE)

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
