### MemGPT + koboldcpp

1. Download + install [koboldcpp](https://github.com/LostRuins/koboldcpp/) and the model you want to test with
2. In your terminal, run `./koboldcpp.py <MODEL> -contextsize <CONTEXT_LENGTH>`

For example, if we downloaded the model `dolphin-2.2.1-mistral-7b.Q6_K.gguf` and put it inside `~/models/TheBloke/`, we would run:
```sh
# using `-contextsize 8192` because Dolphin Mistral 7B has a context length of 8000 (and koboldcpp wants specific intervals, 8192 is the closest)
# the default port is 5001
./koboldcpp.py ~/models/TheBloke/dolphin-2.2.1-mistral-7B-GGUF/dolphin-2.2.1-mistral-7b.Q6_K.gguf --contextsize 8192
```

In your terminal where you're running MemGPT, run:
```sh
export OPENAI_API_BASE=http://localhost:5001
export BACKEND_TYPE=koboldcpp
```
