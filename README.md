<p align="center">
  <a href="https://memgpt.ai"><img src="https://github.com/cpacker/MemGPT/assets/5475622/80f2f418-ef92-4f7a-acab-5d642faa4991" alt="MemGPT logo"></a>
</p>

<div align="center">

 <strong>MemGPT allows you to build LLM agents with long term memory & custom tools</strong>

[![Discord](https://img.shields.io/discord/1161736243340640419?label=Discord&logo=discord&logoColor=5865F2&style=flat-square&color=5865F2)](https://discord.gg/9GEQrxmVyE)
[![Twitter Follow](https://img.shields.io/badge/follow-%40MemGPT-1DA1F2?style=flat-square&logo=x&logoColor=white)](https://twitter.com/MemGPT)
[![arxiv 2310.08560](https://img.shields.io/badge/arXiv-2310.08560-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2310.08560)
[![Documentation](https://img.shields.io/github/v/release/cpacker/MemGPT?label=Documentation&logo=readthedocs&style=flat-square)](https://memgpt.readme.io/docs)

</div>

MemGPT makes it easy to build and deploy stateful LLM agents with support for:
* Long term memory/state management
* Connections to [external data sources](https://memgpt.readme.io/docs/data_sources) (e.g. PDF files) for RAG
* Defining and calling [custom tools](https://memgpt.readme.io/docs/functions) (e.g. [google search](https://github.com/cpacker/MemGPT/blob/main/examples/google_search.py))

You can also use MemGPT to depoy agents as a *service*. You can use a MemGPT server to run a multi-user, multi-agent application on top of supported LLM providers.

<img width="1000" alt="image" src="https://github.com/cpacker/MemGPT/assets/8505980/1096eb91-139a-4bc5-b908-fa585462da09">


## Installation & Setup   
Install MemGPT:
```sh
pip install -U pymemgpt
```

To use MemGPT with OpenAI, set the environment variable `OPENAI_API_KEY` to your OpenAI key then run: 
```
memgpt quickstart --backend openai
```
To use MemGPT with a free hosted endpoint, you run run:
```
memgpt quickstart --backend memgpt
```
For more advanced configuration options or to use a different [LLM backend](https://memgpt.readme.io/docs/endpoints) or [local LLMs](https://memgpt.readme.io/docs/local_llm), run `memgpt configure`.

## Quickstart (CLI)
You can create and chat with a MemGPT agent by running `memgpt run` in your CLI. The `run` command supports the following optional flags (see the [CLI documentation](https://memgpt.readme.io/docs/quickstart) for the full list of flags):
* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

You can view the list of available in-chat commands (e.g. `/memory`, `/exit`) in the [CLI documentation](https://memgpt.readme.io/docs/quickstart).

## Dev portal (alpha build)
MemGPT provides a developer portal that enables you to easily create, edit, monitor, and chat with your MemGPT agents. The easiest way to use the dev portal is to install MemGPT via **docker** (see instructions below).

<img width="1000" alt="image" src="https://github.com/cpacker/MemGPT/assets/5475622/071117c5-46a7-4953-bc9d-d74880e66258">

## Quickstart (Server)  

**Option 1 (Recommended)**: Run with docker compose  
1. [Install docker on your system](https://docs.docker.com/get-docker/)
2. Clone the repo: `git clone https://github.com/cpacker/MemGPT.git`
3. Run `docker compose up`
4. Go to `memgpt.localhost` in the browser to view the developer portal 

**Option 2:** Run with the CLI:
1. Run `memgpt server`
2. Go to `localhost:8283` in the browser to view the developer portal

Once the server is running, you can use the [Python client](https://memgpt.readme.io/docs/admin-client) or [REST API](https://memgpt.readme.io/reference/api) to connect to `memgpt.localhost` (if you're running with docker compose) or `localhost:8283` (if you're running with the CLI) to create users, agents, and more. The service requires authentication with a MemGPT admin password, which can be set with running `export MEMGPT_SERVER_PASS=password`. 

## Supported Endpoints & Backends
MemGPT is designed to be model and provider agnostic. The following LLM and embedding endpoints are supported:

| Provider            | LLM Endpoint    | Embedding Endpoint |
|---------------------|-----------------|--------------------|
| OpenAI              | ✅               | ✅                  |
| Azure OpenAI        | ✅               | ✅                  |
| Google AI (Gemini)  | ✅               | ❌                  |
| Anthropic (Claude)  | ✅               | ❌                  |
| Groq                | ✅ (alpha release) | ❌                  |
| Cohere API          | ✅               | ❌                  |
| vLLM                | ✅               | ❌                  |
| Ollama              | ✅               | ❌                  |
| LM Studio           | ✅               | ❌                  |
| koboldcpp           | ✅               | ❌                  |
| oobabooga web UI    | ✅               | ❌                  |
| llama.cpp           | ✅               | ❌                  |
| HuggingFace TEI     | ❌               | ✅                  |

When using MemGPT with open LLMs (such as those downloaded from HuggingFace), the performance of MemGPT will be highly dependent on the LLM's function calling ability. You can find a list of LLMs/models that are known to work well with MemGPT on the [#model-chat channel on Discord](https://discord.gg/9GEQrxmVyE), as well as on [this spreadsheet](https://docs.google.com/spreadsheets/d/1fH-FdaO8BltTMa4kXiNCxmBCQ46PRBVp3Vn6WbPgsFs/edit?usp=sharing).

## How to Get Involved
* **Contribute to the Project**: Interested in contributing? Start by reading our [Contribution Guidelines](CONTRIBUTING.md).
* **Ask a Question**: Join our community on [Discord](https://discord.gg/9GEQrxmVyE) and direct your questions to the `#support` channel.
* **Report Issues or Suggest Features**: Have an issue or a feature request? Please submit them through our [GitHub Issues page](https://github.com/cpacker/MemGPT/issues).
* **Explore the Roadmap**: Curious about future developments? View and comment on our [project roadmap](https://github.com/cpacker/MemGPT/issues/1200).
* **Benchmark the Performance**: Want to benchmark the performance of a model on MemGPT? Follow our [Benchmarking Guidance](#benchmarking-guidance).
* **Join Community Events**: Stay updated with the [MemGPT event calendar](https://lu.ma/berkeley-llm-meetup) or follow our [Twitter account](https://twitter.com/MemGPT). 


## Benchmarking Guidance
To evaluate the performance of a model on MemGPT, simply configure the appropriate model settings using `memgpt configure`, and then initiate the benchmark via `memgpt benchmark`. The duration will vary depending on your hardware. This will run through a predefined set of prompts through multiple iterations to test the function calling capabilities of a model. You can help track what LLMs work well with MemGPT by contributing your benchmark results via [this form](https://forms.gle/XiBGKEEPFFLNSR348), which will be used to update the spreadsheet.

## Legal notices
By using MemGPT and related MemGPT services (such as the MemGPT endpoint or hosted service), you agree to our [privacy policy](PRIVACY.md) and [terms of service](TERMS.md).

