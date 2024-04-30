<p align="center">
  <a href="https://memgpt.ai"><img src="https://github.com/cpacker/MemGPT/assets/5475622/80f2f418-ef92-4f7a-acab-5d642faa4991" alt="MemGPT logo"></a>
</p>

<div align="center">

 <strong>MemGPT allows you to build LLM agents with self-editing memory</strong>

[![Discord](https://img.shields.io/discord/1161736243340640419?label=Discord&logo=discord&logoColor=5865F2&style=flat-square&color=5865F2)](https://discord.gg/9GEQrxmVyE)
[![arxiv 2310.08560](https://img.shields.io/badge/arXiv-2310.08560-B31B1B?logo=arxiv&style=flat-square)](https://arxiv.org/abs/2310.08560)
[![Documentation](https://img.shields.io/github/v/release/cpacker/MemGPT?label=Documentation&logo=readthedocs&style=flat-square)](https://memgpt.readme.io/docs)

</div>

MemGPT makes it easy to build and deploy stateful LLM agents. With MemGPT, you can build agents with:
* Long term memory/state management
* Connections to external data sources (RAG)
* Defining and calling custom tools (aka. functions)

## Installation & Setup   
Install MemGPT:
```sh
pip install -U pymemgpt
```
To use MemGPT with OpenAI, set the enviornemnt variable `OPENAI_API_KEY` to your OpenAI key then run: 
```
memgpt quickstart --backend openai
```
To use MemGPT with a free hosted endpoint, you run run: 
```
memgpt quickstart --backend memgpt
```
For more advanced configuration options or to use a different LLM backend, run `memgpt configure`. 

## Quickstart (CLI)  
You can create and chat with a MemGPT agent by running `memgpt run` in your CLI. 

## Quickstart (Server)  
You can use MemGPT to depoy agents as a service. You can start a MemGPT *service* in two ways: 

**Option 1 (Recommended)**: Run with docker compose  
1. Clone the repo: `git clone git@github.com:cpacker/MemGPT.git`
2. Run `docker compose up`
3. Go to `memgpt.localhost` in the browser to view the developer portal 

**Option 2:** Run with the CLI:
1. Run `memgpt server`
2. Go to `localhost:8283` in the browser to view the developer portal 

Once the server is running, you can use the REST API to either `memgpt.localhost` (if you're running with docker compose) or `localhost:8283` (if you're running with the CLI) to create users, agents, and more. 

### Python Client 
The Python client can be connected to a running MemGPT service to 

### Python Client 
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

## Documentation
See full documentation at: https://memgpt.readme.io


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

## Legal notices
By using MemGPT and related MemGPT services (such as the MemGPT endpoint or hosted service), you agree to our [privacy policy](PRIVACY.md) and [terms of service](TERMS.md).

## Roadmap
You can view (and comment on!) the MemGPT developer roadmap on GitHub: https://github.com/cpacker/MemGPT/issues/1200.
