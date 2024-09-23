---
title: Using the Letta API
excerpt: How to set up a local Letta API server
category: 658135e7f596b800715c1cee
---

![letta llama](https://raw.githubusercontent.com/cpacker/Letta/main/docs/assets/letta_server.webp)

> ⚠️ API under active development
>
> The Letta API is under **active development** and **changes are being made frequently**.
>
> For support and to track ongoing developments, please visit [the Letta Discord server](https://discord.gg/9GEQrxmVyE) where you can chat with the Letta team and other developers about the API.

Letta can be run as a (multi-user) server process, allowing you to interact with agents using a REST API and use Letta to power your LLM apps.

## Before getting started

To run the Letta server process, you'll need to have already installed and configured Letta (you must have already run `letta configure` or `letta quickstart`).

Before attempting to launch a server process, make sure that you have already configured Letta (using `letta configure`) and are able to successfully create and message an agent using `letta run`. For more information, see [our quickstart guide](https://letta.readme.io/docs/quickstart).

## Starting a server process

You can spawn a Letta server process using the following command:
```sh
letta server
```

If the server was set up correctly, you should see output indicating that the server has been started (by default, the server will listen on `http://localhost:8283`:
```
INFO:     Started server process
INFO:     Waiting for application startup.
Writing out openapi_letta.json file
Writing out openapi_assistants.json file
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8283 (Press CTRL+C to quit)
```

### Using the server admin account

The Letta server will generate a random **admin password** per-session, which will be outputted to your terminal: 
```
Generated admin server password for this session: RHSkTDPkuTMaTTsGq8zIiA
```

This admin password can be used on the **admin routes** (via passing it as a bearer token), which are used to create new users and per-user API keys. 

The admin password can also be also be manually set via the environment variable `MEMGPT_SERVER_PASS`:
```sh
# if MEMGPT_SERVER_PASS is set, the Letta server will use the value as the password instead of randomly generating one
export MEMGPT_SERVER_PASS=ilovellms
```

### Server options

You can modify various server settings via flags to the `letta server command`:

- To run on HTTPS with a self-signed cert, use `--use-ssl`
- To change the port or host, use `--port` and `--host`

To see the full set of option, run `letta server --help`

## Example: Basic usage (using the admin account and default user)

The easiest way to use the Letta API via the Letta server process is to authenticate all REST API calls using the admin password.

When you authenticate REST API calls with the admin password, the server will run all non-admin commands (e.g. creating an agent or sending an agent a message) using the default Letta user, which is the same user that is used when interacting with Letta via the CLI.

In this series of examples, we're assuming we started the server with the admin password `ilovellms`:
```sh
# set the admin password
export MEMGPT_SERVER_PASS=ilovellms
# run the server
letta server
```

### Creating an agent

To create an agent, we can use the [create agent route](https://letta.readme.io/reference/create_agent_api_agents_post):
```sh
curl --request POST \
     --url http://localhost:8283/api/agents \
     --header 'accept: application/json' \
     --header 'authorization: Bearer ilovellms' \
     --header 'content-type: application/json' \
     --data '
{
  "config": {
    "name": "MyCustomAgent",
    "preset": "memgpt_chat",
    "human": "cs_phd",
    "persona": "sam_pov"
  }
}
'
```

This REST call will return the `AgentState` of the newly created agent, which contains its `id` (as well as the `user_id` of the default user):
```
{"agent_state":{"id":"e7a192e6-f9a3-4f60-9e7c-1720d3d207ef","name":"MyCustomAgent","user_id":...
```

### Sending a message to an agent and receiving the reply

To send a message to this agent, we can copy the agent ID from the previous response (`e7a192e6-f9a3-4f60-9e7c-1720d3d207ef`) and use it in a REST call to the [send message route](https://letta.readme.io/reference/send_message_api_agents_message_post).

Let's send the message _"what's the meaning of life? someone told me it's 42..."_:
```sh
curl --request POST \
     --url http://localhost:8283/api/agents/<agent_id>/messages \
     --header 'accept: application/json' \
     --header 'authorization: Bearer ilovellms' \
     --header 'content-type: application/json' \
     --data @- <<EOF
{
  "agent_id": "e7a192e6-f9a3-4f60-9e7c-1720d3d207ef",
  "message": "what's the meaning of life? someone told me it's 42...",
  "stream": true,
  "role": "user"
}
EOF
```

Our response will stream back and look like this:
```sh
data: {"internal_monologue": "A fascinating question. It seems Chad may be referencing \"The Hitchhiker's Guide to the Galaxy\" with that number, 42. How should I respond to this thoughtful query, I wonder? Engage him with philosophical discourse or humorous banter? Maybe a mix of both would be most suitable. After all, life is about balance. Let's craft a response...", "date": "2024-02-29T06:07:47.844138+00:00"}

data: {"function_call": "send_message({'message': \"Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?\"})", "date": "2024-02-29T06:07:48.844733+00:00"}

data: {"assistant_message": "Ah, the age-old question, Chad. The meaning of life is as subjective as the life itself. 42, as the supercomputer 'Deep Thought' calculated in 'The Hitchhiker's Guide to the Galaxy', is indeed an answer, but maybe not the one we're after. Among other things, perhaps life is about learning, experiencing and connecting. What are your thoughts, Chad? What gives your life meaning?", "date": "2024-02-29T06:07:49.846280+00:00"}

data: {"function_return": "None", "status": "success", "date": "2024-02-29T06:07:50.847262+00:00"}
```

## Example: Multi-user setup

In settings where you want to use the Letta server to power a multi-user application (e.g. a chatbot service), you'll likely want to have separate users, each with their own library of agents.

To handle the setting with multiple users, you can use the admin routes to create users and generate per-user API keys.

Once you have a user's API key, simply pass the API key via the bearer token to the non-admin routes, and the API call will be associated with the user that owns the API key.

### Creating a user

Let's create a new user and get their API key. To do so, we can use the [create user route](https://letta.readme.io/reference/create_user_admin_users_post):
```sh
curl --request POST \
     --url http://localhost:8283/admin/users \
     --header 'accept: application/json' \
     --header 'authorization: Bearer ilovellms' \
     --header 'content-type: application/json' \
     --data '{}'
```

The response back provides the `id` of the new user, as well as an API key for that user which we'll use to associate API calls with the user profile:
```sh
{"user_id":"26fd194b-a34e-4ba5-a8e5-0e626f439962","api_key":"sk-0e992ddd0...94656a7ddf6"}%
```

Now when we make calls to `http://localhost:8283/api`, we can pass in `sk-0e992ddd0...94656a7ddf6` as the bearer token to associate an agent calls with a specific user.
