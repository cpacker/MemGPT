1. Download + install [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html) and the model you want to test with
2. Launch a vLLM API server using [the official vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)

For example, if we downloaded the model `dolphin-2.2.1-mistral-7b.Q6_K.gguf` and put it inside `~/models/TheBloke/`, we would run:
```sh
python -m vllm.entrypoints.openai.api_server \
--model dolphin-2.2.1-mistral-7b.Q6_K.gguf 
```

In your terminal where you're running MemGPT, run `memgpt configure` to set the default backend for MemGPT to point at vLLM:
```
# if you are running vLLM locally, the default IP address + port will be http://localhost:8000
? Select LLM inference provider: local
? Select LLM backend (select 'openai' if you have an OpenAI compatible proxy): vllm
? Enter default endpoint: http://localhost:8000
? Enter HuggingFace model tag (e.g. ehartford/dolphin-2.2.1-mistral-7b):
...
```

If you have an existing agent that you want to move to the vLLM backend, add extra flags to `memgpt run`:
```sh
memgpt run --agent your_agent --model-endpoint-type vLLM --model-endpoint http://localhost:8000
```