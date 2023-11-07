### MemGPT + LM Studio

![image](https://github.com/cpacker/MemGPT/assets/5475622/abc8ce2d-4130-4c51-8169-83e682db625d)

1. Download [LM Studio](https://lmstudio.ai/) and the model you want to test with
2. Go to the "local inference server" tab, load the model and configure your settings (make sure to set the context length to something reasonable like 8k!)
3. Click "Start server"
4. Copy the IP address + port that your server is running on (in the example screenshot, the address is `http://localhost:1234`)

In your terminal where you're running MemGPT, run:

```sh
# if you used a different port in LM Studio, change 1234 to the actual port
export OPENAI_API_BASE=http://localhost:1234
export BACKEND_TYPE=lmstudio
```

* **Make sure that "context length" is set to the max context length of the model you're using (e.g. 8000 for Mistral 7B variants)**
* **If you see "Prompt Formatting" in your menu, turn it off**