### MemGPT + Ollama

!!! warning "Be careful when downloading Ollama models!"

    Make sure to use tags when downloading Ollama models! Don't do `ollama run dolphin2.2-mistral`, do `ollama run dolphin2.2-mistral:7b-q6_K`.

    If you don't specify a tag, Ollama may default to using a highly compressed model variant (e.g. Q4). We highly recommend **NOT** using a compression level below Q4 (stick to Q6, Q8, or fp16 if possible). In our testing, models below Q6 start to become extremely unstable when used with MemGPT.

1. Download + install [Ollama](https://github.com/jmorganca/ollama) and the model you want to test with
2. Download a model to test with by running `ollama run <MODEL_NAME>` in the terminal (check the [Ollama model library](https://ollama.ai/library) for available models)
3. In addition to setting `OPENAI_API_BASE` and `BACKEND_TYPE`, we additionally need to set `OLLAMA_MODEL` (to the Ollama model name)

For example, if we want to use Dolphin 2.2.1 Mistral, we can download it by running:
```sh
# Let's use the q6_K variant
ollama run dolphin2.2-mistral:7b-q6_K
```
```text
pulling manifest
pulling d8a5ee4aba09... 100% |█████████████████████████████████████████████████████████████████████████| (4.1/4.1 GB, 20 MB/s)
pulling a47b02e00552... 100% |██████████████████████████████████████████████████████████████████████████████| (106/106 B, 77 B/s)
pulling 9640c2212a51... 100% |████████████████████████████████████████████████████████████████████████████████| (41/41 B, 22 B/s)
pulling de6bcd73f9b4... 100% |████████████████████████████████████████████████████████████████████████████████| (58/58 B, 28 B/s)
pulling 95c3d8d4429f... 100% |█████████████████████████████████████████████████████████████████████████████| (455/455 B, 330 B/s)
verifying sha256 digest
writing manifest
removing any unused layers
success
```

In your terminal where you're running MemGPT, run:
```sh
# By default, Ollama runs an API server on port 11434
export OPENAI_API_BASE=http://localhost:11434
export BACKEND_TYPE=ollama

# Make sure to add the tag!
export OLLAMA_MODEL=dolphin2.2-mistral:7b-q6_K
```
