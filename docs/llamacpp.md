### MemGPT + llama.cpp

1. Download + install [llama.cpp](https://github.com/ggerganov/llama.cpp) and the model you want to test with
2. In your terminal, run `./server -m <MODEL> -c <CONTEXT_LENGTH>`

For example, if we downloaded the model `dolphin-2.2.1-mistral-7b.Q6_K.gguf` and put it inside `~/models/TheBloke/`, we would run:
```sh
# using `-c 8000` because Dolphin Mistral 7B has a context length of 8000
# the default port is 8080, you can change this with `--port`
./server -m ~/models/TheBloke/dolphin-2.2.1-mistral-7B-GGUF/dolphin-2.2.1-mistral-7b.Q6_K.gguf -c 8000
```

In your terminal where you're running MemGPT, run:
```sh
export OPENAI_API_BASE=http://localhost:8080
export BACKEND_TYPE=llamacpp
```
