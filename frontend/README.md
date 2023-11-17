
Chat UI frontend for MemGPT, based on [LlamaIndex Chat](https://github.com/run-llama/chat-llamaindex), which itself is based on [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web).

## ⚡️ Quick start

### Install and start the local frontend

Requirement: [NodeJS](https://nodejs.org) 18

- Navigate to the frontend directory

```bash
cd frontend
```

- Run the dev server

```bash
pnpm install
pnpm dev
```

### Start the websocket server

```sh
python memgpt/server/websocket_server.py
```
