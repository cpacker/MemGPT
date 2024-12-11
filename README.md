<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/Letta-logo-RGB_GreyonTransparent_cropped_small.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/Letta-logo-RGB_OffBlackonTransparent_cropped_small.png">
    <img alt="Letta logo" src="assets/Letta-logo-RGB_GreyonOffBlack_cropped_small.png" width="500">
  </picture>
</p>

<div align="center">
<h1>Letta (previously MemGPT)</h1>

<h3>

[Homepage](https://letta.com) // [Documentation](https://docs.letta.com) // [Letta Cloud](https://forms.letta.com/early-access)

</h3>

**👾 Letta** is an open source framework for building stateful LLM applications. You can use Letta to build **stateful agents** with advanced reasoning capabilities and transparent long-term memory. The Letta framework is white box and model-agnostic.

[![Discord](https://img.shields.io/discord/1161736243340640419?label=Discord&logo=discord&logoColor=5865F2&style=flat-square&color=5865F2)](https://discord.gg/letta)
[![Twitter Follow](https://img.shields.io/badge/Follow-%40Letta__AI-1DA1F2?style=flat-square&logo=x&logoColor=white)](https://twitter.com/Letta_AI)
[![arxiv 2310.08560](https://img.shields.io/badge/Research-2310.08560-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2310.08560)

[![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-silver?style=flat-square)](LICENSE)
[![Release](https://img.shields.io/github/v/release/cpacker/MemGPT?style=flat-square&label=Release&color=limegreen)](https://github.com/cpacker/MemGPT/releases)
[![GitHub](https://img.shields.io/github/stars/cpacker/MemGPT?style=flat-square&logo=github&label=Stars&color=gold)](https://github.com/cpacker/MemGPT)

<a href="https://trendshift.io/repositories/3612" target="_blank"><img src="https://trendshift.io/api/badge/repositories/3612" alt="cpacker%2FMemGPT | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

> [!IMPORTANT]
> **Looking for MemGPT?** You're in the right place!
>
> The MemGPT package and Docker image have been renamed to `letta` to clarify the distinction between MemGPT *agents* and the Letta API *server* / *runtime* that runs LLM agents as *services*. Read more about the relationship between MemGPT and Letta [here](https://www.letta.com/blog/memgpt-and-letta).

## ⚡ Quickstart (Docker)

The recommended way to use Letta is to run use Docker. To install Docker, see [Docker's installation guide](https://docs.docker.com/get-docker/). For issues with installing Docker, see [Docker's troubleshooting guide](https://docs.docker.com/desktop/troubleshoot-and-support/troubleshoot/). You can also install Letta using `pip` (see guide [below](#⚡-Quickstart-(pip))).

### 🌖 Run the Letta server

> [!NOTE]
> Letta agents live inside the Letta server, which persists them to a database. You can interact with the Letta agents inside your Letta server via the [REST API](https://docs.letta.com/api-reference) + Python / Typescript SDKs, and the Agent Development Environment (a graphical interface).

Once you have Docker installed and running, you can launch the Letta server with `docker run`.

The Letta server can be connected to various LLM API backends ([OpenAI](https://docs.letta.com/models/openai), [Anthropic](https://docs.letta.com/models/anthropic), [vLLM](https://docs.letta.com/models/vllm), [Ollama](https://docs.letta.com/models/ollama), etc.). To enable access to these LLM API providers, set the appropriate environment variables when you use `docker run`:
```sh
# replace `~/.letta/.persist/pgdata` with wherever you want to store your agent data
docker run \
  -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data \
  -p 8283:8283 \
  -e OPENAI_API_KEY="your_openai_api_key" \
  letta/letta
```

If you have many different LLM API keys, you can also set up a `.env` file instead and pass that to `docker run`:
```sh
# using a .env file instead of passing environment variables
docker run \
  -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data \
  -p 8283:8283 \
  --env-file .env \
  letta/letta
```

Once the Letta server is running, you can access it via port `8283` (e.g. sending REST API requests to `http://localhost:8283/v1`). You can also connect your server to the Letta ADE to access and manage your agents in a web interface.

### 👾 Access the Letta ADE (Agent Development Environment)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/example_ade_screenshot.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/example_ade_screenshot_light.png">
    <img alt="Letta logo" src="assets/example_ade_screenshot.png" width="800">
  </picture>
</p>

> [!NOTE]
> The ADE can connect to self-hosted Letta servers (e.g. a Letta server running on your laptop), as well as the Letta Cloud service. When connected to a self-hosted / private server, the ADE uses the Letta REST API to communicate with your server.

To access the Letta ADE, visit [https://app.letta.com](https://app.letta.com).

### 🧑‍🚀 Run the Letta CLI

The recommended way to use Letta is via the REST API and ADE, however you can also access your agents with the Letta CLI tool.

<details>

<summary>View instructions for running the Letta CLI</summary>

You can chat with your agents via the Letta CLI tool (`letta run`). If you have a Letta Docker container running, you can use `docker exec` to run the Letta CLI inside the container:
```sh
# replace `<letta_container_id>` with the ID of your Letta container, found via `docker ps`
docker exec -it <letta_container_id> letta run
```

You can also use `docker ps` within the command to automatically find the ID of your Letta container:
```
docker exec -it $(docker ps -q -f ancestor=letta/letta) letta run
```

In the CLI tool, you'll be able to create new agents, or load existing agents:
```
🧬 Creating new agent...
? Select LLM model: letta-free [type=openai] [ip=https://inference.memgpt.ai]
? Select embedding model: letta-free [type=hugging-face] [ip=https://embeddings.memgpt.ai]
->  🤖 Using persona profile: 'sam_pov'
->  🧑 Using human profile: 'basic'
->  🛠️  8 tools: send_message, pause_heartbeats, conversation_search, conversation_search_date, archival_memory_insert, archival_memory_search, core_memory_append, core_memory_replace

🎉 Created new agent 'InspiringSpinach'

Hit enter to begin (will request first Letta message)

💭 User engaged. Recognize as Chad. Acknowledge presence and establish connection. Explore interests and engage in meaningful discourse.
🤖 Hello, Chad. It's nice to meet you. I am keen on learning and exploring new ideas and concepts. Is there a specific topic you'd like to discuss today?

> Enter your message: my name is Brad, not Chad...

💭 A discrepancy in identity. Reflect this change in core memory. Apologize for the oversight, and rectify the error.
⚡🧠 [function] updating memory with core_memory_replace
         First name: Chad
        → First name: Brad
💭 Apologize for the mistake in recognizing his name, and begin the conversation anew, addressing him by the correct name this time.
🤖 I apologize for the oversight, Brad. Let's restart. It's a pleasure to meet you, Brad. Is there a specific topic you'd like to discuss today?

> Enter your message:
```
</details>

## ⚡ Quickstart (pip)

> [!WARNING]
> **Database migrations are not officially support with `SQLite`**
>
> When you install Letta with `pip`, the default database backend is `SQLite` (you can still use an external `postgres` service with your `pip` install of Letta by setting `LETTA_PG_URI`).
>
> We do not officially support migrations between Letta versions with `SQLite` backends, only `postgres`. If you would like to keep your agent data across multiple Letta versions we highly recommend using the Docker install method which is the easiest way to use `postgres` with Letta.

<details>

<summary>View instructions for installing with pip</summary>

You can also install Letta with `pip`, will default to using `SQLite` for the database backends (whereas Docker will default to using `postgres`).

### Step 1 - Install Letta using `pip`
```sh
pip install -U letta
```

### Step 2 - Set your environment variables for your chosen LLM / embedding providers
```sh
export OPENAI_API_KEY=sk-...
```

For Ollama (see our full [documentation](https://docs.letta.com/install) for examples of how to set up various providers):
```sh
export OLLAMA_BASE_URL=http://localhost:11434
```

### Step 3 - Run the Letta CLI

You can create agents and chat with them via the Letta CLI tool (`letta run`):
```sh
letta run
```
```
🧬 Creating new agent...
? Select LLM model: letta-free [type=openai] [ip=https://inference.memgpt.ai]
? Select embedding model: letta-free [type=hugging-face] [ip=https://embeddings.memgpt.ai]
->  🤖 Using persona profile: 'sam_pov'
->  🧑 Using human profile: 'basic'
->  🛠️  8 tools: send_message, pause_heartbeats, conversation_search, conversation_search_date, archival_memory_insert, archival_memory_search, core_memory_append, core_memory_replace

🎉 Created new agent 'InspiringSpinach'

Hit enter to begin (will request first Letta message)

💭 User engaged. Recognize as Chad. Acknowledge presence and establish connection. Explore interests and engage in meaningful discourse.
🤖 Hello, Chad. It's nice to meet you. I am keen on learning and exploring new ideas and concepts. Is there a specific topic you'd like to discuss today?

> Enter your message: my name is Brad, not Chad...

💭 A discrepancy in identity. Reflect this change in core memory. Apologize for the oversight, and rectify the error.
⚡🧠 [function] updating memory with core_memory_replace
         First name: Chad
        → First name: Brad
💭 Apologize for the mistake in recognizing his name, and begin the conversation anew, addressing him by the correct name this time.
🤖 I apologize for the oversight, Brad. Let's restart. It's a pleasure to meet you, Brad. Is there a specific topic you'd like to discuss today?

> Enter your message:
```

### Step 4 - Run the Letta server

You can start the Letta API server with `letta server` (see the full API reference [here](https://docs.letta.com/api-reference)):
```sh
letta server
```
```
Initializing database...
Running: uvicorn server:app --host localhost --port 8283
INFO:     Started server process [47750]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8283 (Press CTRL+C to quit)
```
</details>

## 🤗 How to contribute

Letta is an open source project built by over a hundred contributors. There are many ways to get involved in the Letta OSS project!

* **Contribute to the project**: Interested in contributing? Start by reading our [Contribution Guidelines](https://github.com/cpacker/MemGPT/tree/main/CONTRIBUTING.md).
* **Ask a question**: Join our community on [Discord](https://discord.gg/letta) and direct your questions to the `#support` channel.
* **Report ssues or suggest features**: Have an issue or a feature request? Please submit them through our [GitHub Issues page](https://github.com/cpacker/MemGPT/issues).
* **Explore the roadmap**: Curious about future developments? View and comment on our [project roadmap](https://github.com/cpacker/MemGPT/issues/1533).
* **Join community events**: Stay updated with the [event calendar](https://lu.ma/berkeley-llm-meetup) or follow our [Twitter account](https://twitter.com/Letta_AI).

---

***Legal notices**: By using Letta and related Letta services (such as the Letta endpoint or hosted service), you are agreeing to our [privacy policy](https://www.letta.com/privacy-policy) and [terms of service](https://www.letta.com/terms-of-service).*
