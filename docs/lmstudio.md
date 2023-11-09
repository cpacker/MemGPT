### MemGPT + LM Studio

!!! warning "Important LM Studio settings"

    Make sure that "context length" is set (inside LM Studio's "Model Configuration" panel) to the max context length of the model you're using (e.g. 8000 for Mistral 7B variants).

    If you see "Prompt Formatting" (inside LM Studio's "Server Options" panel), turn it **OFF**. Leaving it **ON** will break MemGPT.

![image](https://github.com/cpacker/MemGPT/assets/5475622/74fd5e4d-a549-482d-b9f5-44b1829f41a8)

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
