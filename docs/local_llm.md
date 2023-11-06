## Using MemGPT with local LLMs

### ⁉️ IMPORTANT

When using open LLMs with MemGPT, **the main failure case will be your LLM outputting a string that cannot be understood by MemGPT**. MemGPT uses function calling to manage memory (eg `edit_core_memory(...)` and interact with the user (`send_message(...)`), so your LLM needs generate outputs that can be parsed into MemGPT function calls.

### ⚡ Quick overview

1. Put your own LLM behind a web server API (e.g. [oobabooga web UI](https://github.com/oobabooga/text-generation-webui#starting-the-web-ui))
2. Set `OPENAI_API_BASE=YOUR_API_IP_ADDRESS` and `BACKEND_TYPE=webui`
3. Run MemGPT with `python3 main.py --no_verify`, it should now use your LLM instead of OpenAI GPT
4. If things aren't working, read the full instructions below

###