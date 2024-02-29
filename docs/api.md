---
title: Using the MemGPT API
excerpt: How to set up a local MemGPT API server
category: 658135e7f596b800715c1cee
---

![memgpt llama](https://raw.githubusercontent.com/cpacker/MemGPT/main/docs/assets/memgpt_server.webp)

> ⚠️ API under active development
>
> The MemGPT API is under **active development** and **changes are being made frequently**.
>
> For support and to track ongoing developments, please visit [the MemGPT Discord server](https://discord.gg/9GEQrxmVyE) where you can chat with the MemGPT team and other developers about the API.

MemGPT can be run as a (multi-user) server process, allowing you to interact with agents using a REST API and use MemGPT to power your LLM apps.

## Before getting started

To run the MemGPT server process, you'll need to have already installed and configured MemGPT (you must have already run `memgpt configure` or `memgpt quickstart`).

Before attempting to launch a server process, make sure that you have already configured MemGPT (using `memgpt configure`) and are able to successfully create and message an agent using `memgpt run`. For more information, see [our quickstart guide](https://memgpt.readme.io/docs/quickstart).

## Starting a server process

You can spawn a MemGPT server process using the following command:
```sh
memgpt server
```

If the server was set up correctly, you should see output indicating that the server has been started (by default, the server will listen on `http://localhost:8283`:
```
INFO:     Started server process
INFO:     Waiting for application startup.
Writing out openapi_memgpt.json file
Writing out openapi_assistants.json file
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8283 (Press CTRL+C to quit)
```

### Using the server admin account

The MemGPT server will generate a random **admin password** per-session, which will be outputted to your terminal: 
```
Generated admin server password for this session: RHSkTDPkuTMaTTsGq8zIiA
```

This admin password can be used on the **admin routes** (via passing it as a bearer token), which are used to create new users and per-user API keys. 

The admin password can also be also be manually set via the environment variable `MEMGPT_SERVER_PASS`:
```sh
# if MEMGPT_SERVER_PASS is set, the MemGPT server will use the value as the password instead of randomly generating one
export MEMGPT_SERVER_PASS=ilovellms
```

### Server options

You can modify various server settings via flags to the `memgpt server command`:

- To run on HTTPS with a self-signed cert, use `--use-ssl`
- To change the port or host, use `--port` and `--host`

To see the full set of option, run `memgpt server --help`

## Example: basic usage

### Creating a user

Examples REST API code, both CURL and python requests

### Creating an agent

Examples REST API code, both CURL and python requests

### Sending a message to an agent and receiving the reply

Examples REST API code, both CURL and python requests
