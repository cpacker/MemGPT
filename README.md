<p align="center">
  <a href="https://letta.ai"><img src="https://github.com/cpacker/Letta/assets/5475622/80f2f418-ef92-4f7a-acab-5d642faa4991" alt="Letta logo"></a>
</p>

<div align="center">

 <strong>Letta allows you to build LLM agents with long term memory & custom tools</strong>

[![Discord](https://img.shields.io/discord/1161736243340640419?label=Discord&logo=discord&logoColor=5865F2&style=flat-square&color=5865F2)](https://discord.gg/9GEQrxmVyE)
[![Twitter Follow](https://img.shields.io/badge/follow-%40Letta-1DA1F2?style=flat-square&logo=x&logoColor=white)](https://twitter.com/Letta)
[![arxiv 2310.08560](https://img.shields.io/badge/arXiv-2310.08560-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2310.08560)
[![Documentation](https://img.shields.io/github/v/release/cpacker/Letta?label=Documentation&logo=readthedocs&style=flat-square)](https://letta.readme.io/docs)

</div>

Letta makes it easy to build and deploy stateful LLM agents with support for:
* Long term memory/state management
* Connections to [external data sources](https://letta.readme.io/docs/data_sources) (e.g. PDF files) for RAG
* Defining and calling [custom tools](https://letta.readme.io/docs/functions) (e.g. [google search](https://github.com/cpacker/Letta/blob/main/examples/google_search.py))

You can also use Letta to deploy agents as a *service*. You can use a Letta server to run a multi-user, multi-agent application on top of supported LLM providers.

<img width="1000" alt="image" src="https://github.com/cpacker/Letta/assets/8505980/1096eb91-139a-4bc5-b908-fa585462da09">


## Installation & Setup
Install Letta:
```sh
pip install -U pyletta
```

To use Letta with OpenAI, set the environment variable `OPENAI_API_KEY` to your OpenAI key then run:
```
letta quickstart --backend openai
```
To use Letta with a free hosted endpoint, you run run:
```
letta quickstart --backend letta
```
For more advanced configuration options or to use a different [LLM backend](https://letta.readme.io/docs/endpoints) or [local LLMs](https://letta.readme.io/docs/local_llm), run `letta configure`.

## Quickstart (CLI)
You can create and chat with a Letta agent by running `letta run` in your CLI. The `run` command supports the following optional flags (see the [CLI documentation](https://letta.readme.io/docs/quickstart) for the full list of flags):
* `--agent`: (str) Name of agent to create or to resume chatting with.
* `--first`: (str) Allow user to sent the first message.
* `--debug`: (bool) Show debug logs (default=False)
* `--no-verify`: (bool) Bypass message verification (default=False)
* `--yes`/`-y`: (bool) Skip confirmation prompt and use defaults (default=False)

You can view the list of available in-chat commands (e.g. `/memory`, `/exit`) in the [CLI documentation](https://letta.readme.io/docs/quickstart).

## Dev portal (alpha build)
Letta provides a developer portal that enables you to easily create, edit, monitor, and chat with your Letta agents. The easiest way to use the dev portal is to install Letta via **docker** (see instructions below).

<img width="1000" alt="image" src="https://github.com/cpacker/Letta/assets/5475622/071117c5-46a7-4953-bc9d-d74880e66258">

## Quickstart (Server)

**Option 1 (Recommended)**: Run with docker compose
1. [Install docker on your system](https://docs.docker.com/get-docker/)
2. Clone the repo: `git clone https://github.com/cpacker/Letta.git`
3. Copy-paste `.env.example` to `.env` and optionally modify
4. Run `docker compose up`
5. Go to `letta.localhost` in the browser to view the developer portal

**Option 2:** Run with the CLI:
1. Run `letta server`
2. Go to `localhost:8283` in the browser to view the developer portal

Once the server is running, you can use the [Python client](https://letta.readme.io/docs/admin-client) or [REST API](https://letta.readme.io/reference/api) to connect to `letta.localhost` (if you're running with docker compose) or `localhost:8283` (if you're running with the CLI) to create users, agents, and more. The service requires authentication with a Letta admin password; it is the value of `MEMGPT_SERVER_PASS` in `.env`.

## Supported Endpoints & Backends
Letta is designed to be model and provider agnostic. The following LLM and embedding endpoints are supported:

| Provider            | LLM Endpoint    | Embedding Endpoint |
|---------------------|-----------------|--------------------|
| OpenAI              | ✅               | ✅                  |
| Azure OpenAI        | ✅               | ✅                  |
| Google AI (Gemini)  | ✅               | ❌                  |
| Anthropic (Claude)  | ✅               | ❌                  |
| Groq                | ✅ (alpha release) | ❌                  |
| Cohere API          | ✅               | ❌                  |
| vLLM                | ✅               | ❌                  |
| Ollama              | ✅               | ✅                  |
| LM Studio           | ✅               | ❌                  |
| koboldcpp           | ✅               | ❌                  |
| oobabooga web UI    | ✅               | ❌                  |
| llama.cpp           | ✅               | ❌                  |
| HuggingFace TEI     | ❌               | ✅                  |

When using Letta with open LLMs (such as those downloaded from HuggingFace), the performance of Letta will be highly dependent on the LLM's function calling ability. You can find a list of LLMs/models that are known to work well with Letta on the [#model-chat channel on Discord](https://discord.gg/9GEQrxmVyE), as well as on [this spreadsheet](https://docs.google.com/spreadsheets/d/1fH-FdaO8BltTMa4kXiNCxmBCQ46PRBVp3Vn6WbPgsFs/edit?usp=sharing).

## How to Get Involved
* **Contribute to the Project**: Interested in contributing? Start by reading our [Contribution Guidelines](https://github.com/cpacker/Letta/tree/main/CONTRIBUTING.md).
* **Ask a Question**: Join our community on [Discord](https://discord.gg/9GEQrxmVyE) and direct your questions to the `#support` channel.
* **Report Issues or Suggest Features**: Have an issue or a feature request? Please submit them through our [GitHub Issues page](https://github.com/cpacker/Letta/issues).
* **Explore the Roadmap**: Curious about future developments? View and comment on our [project roadmap](https://github.com/cpacker/Letta/issues/1200).
* **Benchmark the Performance**: Want to benchmark the performance of a model on Letta? Follow our [Benchmarking Guidance](#benchmarking-guidance).
* **Join Community Events**: Stay updated with the [Letta event calendar](https://lu.ma/berkeley-llm-meetup) or follow our [Twitter account](https://twitter.com/Letta).


## Benchmarking Guidance
To evaluate the performance of a model on Letta, simply configure the appropriate model settings using `letta configure`, and then initiate the benchmark via `letta benchmark`. The duration will vary depending on your hardware. This will run through a predefined set of prompts through multiple iterations to test the function calling capabilities of a model. You can help track what LLMs work well with Letta by contributing your benchmark results via [this form](https://forms.gle/XiBGKEEPFFLNSR348), which will be used to update the spreadsheet.

## Legal notices
By using Letta and related Letta services (such as the Letta endpoint or hosted service), you agree to our [privacy policy](https://github.com/cpacker/Letta/tree/main/PRIVACY.md) and [terms of service](https://github.com/cpacker/Letta/tree/main/TERMS.md).
