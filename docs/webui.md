### MemGPT + web UI

!!! warning "Important web UI settings"

    If you have problems getting web UI set up, please use the [official web UI repo for support](https://github.com/oobabooga/text-generation-webui)! There will be more answered questions about web UI there vs here on the MemGPT repo.

    Do **NOT** enable any extensions in web UI, including the [openai extension](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)! Just run web UI as-is, unless you are running [MemGPT+Autogen](https://github.com/cpacker/MemGPT/tree/main/memgpt/autogen) with non-MemGPT agents.

To get MemGPT to work with a local LLM, you need to have the LLM running on a server that takes API requests.

For the purposes of this example, we're going to serve (host) the LLMs using [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui), but if you want to use something else you can! This also assumes your running web UI locally - if you're running on e.g. Runpod, you'll want to follow Runpod specific instructions (for example use [TheBloke's one-click UI and API](https://github.com/TheBlokeAI/dockerLLM/blob/main/README_Runpod_LocalLLMsUIandAPI.md))

1. Install oobabooga web UI using the instructions [here](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui)
2. Once installed, launch the web server with `python server.py`
3. Navigate to the web app (if local, this is probably [`http://127.0.0.1:7860`](http://localhost:7860)), select the model you want to use, adjust your GPU and CPU memory settings, and click "load"
4. If the model was loaded successfully, you should be able to access it via the API (if local, this is probably on port `5000`)
5. Assuming steps 1-4 went correctly, the LLM is now properly hosted on a port you can point MemGPT to!

In your terminal where you're running MemGPT, run `memgpt configure` to set the default backend for MemGPT to point at web UI:
```
# if you are running web UI locally, the default IP address + port will be http://localhost:5000
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): webui
? Enter default endpoint: http://localhost:5000
...
```

If you have an existing agent that you want to move to the web UI backend, add extra flags to `memgpt run`:
```sh
memgpt run --agent your_agent --model-endpoint-type webui --model-endpoint http://localhost:5000
```

Text gen web UI exposes a lot of parameters that can dramatically change LLM outputs, to change these you can modify the [web UI settings file](https://github.com/cpacker/MemGPT/blob/main/memgpt/local_llm/webui/settings.py).
