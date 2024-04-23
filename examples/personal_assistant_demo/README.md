## Personal assistant demo

In this example we'll create an agent preset that has access to:
1. Gmail (can read your email)
2. Google Calendar (can schedule events)
3. SMS (can text you a message)

# Initial setup

For the Google APIs:
```sh
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

For the Twilio API + listener:
```sh
# Outbound API requests
pip install --upgrade twilio
# Listener
TODO
```

## Setting up the Google APIs

TODO

## Setting up the Twilio API

TODO

# Creating the agent preset

## Create a custom persona

TODO


## Linking the functions

The preset (shown below) and functions are provided for you, so you just need to copy/link them.

```sh
cp examples/personal_assistant_demo/google_calendar.py ~/.memgpt/functions/
cp examples/personal_assistant_demo/twilio_messaging.py ~/.memgpt/functions/
```

(or use the dev portal)

## Creating the preset

```yaml
system_prompt: "memgpt_chat"
functions:
  - "send_message"
  - "pause_heartbeats"
  - "core_memory_append"
  - "core_memory_replace"
  - "conversation_search"
  - "conversation_search_date"
  - "archival_memory_insert"
  - "archival_memory_search"
  - "schedule_event"
  - "send_text_message"
```

```sh
memgpt add preset -f examples/personal_assistant_demo/personal_assistant_preset.yaml --name pa_preset
```

## Creating an agent with the preset

Now we should be able to create an agent with the preset. Make sure to record the `agent_id`.

```sh
TODO
```

# Running the agent with Gmail + SMS listeners

The MemGPT agent can send outbound SMS messages and schedule events with the new tools `send_text_message` and `schedule_event`, but we also want messages to be sent to the agent when:
1. A new email arrives in our inbox
2. An SMS is sent to the phone number used by the agent

## Running the Gmail listener

```sh
TODO
```

## Running the Twilio listener

```sh
# start the python flask server
# then run ngrok
# then update the twilio api page to reflect the ngroq IP
```



## Example conversation

TODO