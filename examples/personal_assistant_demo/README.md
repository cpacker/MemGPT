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

### Setup authentication for Google Calendar

Copy the credentials file to `~/.memgpt/google_api_credentials.json`. Then, run the initial setup script that will take you to a login page:
```sh
python examples/personal_assistant_demo/google_calendar_test_setup.py
```
```
Please visit this URL to authorize this application: https://accounts.google.com/o/oauth2/auth?response_type=...
Getting the upcoming 10 events
2024-04-23T09:00:00-07:00 ...
```

### Setup authentication for Gmail

Similar flow, run the authentication script to generate the token:
```sh
python examples/personal_assistant_demo/gmail_test_setup.py 
```
```
Please visit this URL to authorize this application: https://accounts.google.com/o/oauth2/auth?response_type=...
Labels:
CHAT
SENT
INBOX
IMPORTANT
TRASH
...
```

## Setting up the Twilio API

TODO
Create a Twilio account and set the following variables:
```sh
export TWILIO_ACCOUNT_SID=...
export TWILIO_ACCOUNT_TOKEN=...
export TWILIO_FROM_NUMBER=...
export TWILIO_TO_NUMBER=...
```

# Creating the agent preset

## Create a custom user 

In the demo we'll show how MemGPT can programatically update its knowledge about you:
```
This is what I know so far about the user, I should expand this as I learn more about them.

Name: Charles Packer
Gender: Male
Occupation: CS PhD student working on an AI project with collaborator Sarah Wooders

Notes about their preferred communication style + working habits:
- wakes up at around 7am
- enjoys using (and receiving!) emojis in messages, especially funny combinations of emojis
- prefers sending and receiving shorter messages
- does not like "robotic" sounding assistants, e.g. assistants that say "How can I assist you today?"
```

```sh
memgpt add human -f examples/personal_assistant_demo/charles.txt --name charles
```

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

Now we should be able to create an agent with the preset. Make sure to record the `agent_id`:

```sh
memgpt run --preset pa_preset --persona sam_pov --human charles --stream
```
```
? Would you like to select an existing agent? No

🧬 Creating new agent...
->  🤖 Using persona profile: 'sam_pov'
->  🧑 Using human profile: 'basic'
🎉 Created new agent 'DelicateGiraffe' (id=4c4e97c9-ad8e-4065-b716-838e5d6f7f7b)

Hit enter to begin (will request first MemGPT message)


💭 Unprecedented event, Charles logged into the system for the first time. Warm welcome would set a positive
tone for our future interactions. Don't forget the emoji, he appreciates those little gestures.
🤖 Hello Charles! 👋 Great to have you here. I've been looking forward to our conversations! 😄
```

```sh
AGENT_ID="4c4e97c9-ad8e-4065-b716-838e5d6f7f7b"
```

# Running the agent with Gmail + SMS listeners

The MemGPT agent can send outbound SMS messages and schedule events with the new tools `send_text_message` and `schedule_event`, but we also want messages to be sent to the agent when:
1. A new email arrives in our inbox
2. An SMS is sent to the phone number used by the agent

## Running the Gmail listener

Start the Gmail listener (this will send "new email" updates to the MemGPT server when a new email arrives):
```sh
python examples/personal_assistant_demo/twilio_flask_listener.py $AGENT_ID
```

## Running the Twilio listener

Start the Python Flask server (this will send "new SMS" updates to the MemGPT server when a new SMS arrives):
```sh
python examples/personal_assistant_demo/twilio_flask_listener.py $AGENT_ID
```

Run `ngrok` to expose your local Flask server to a public IP (Twilio will POST to this server when an inbound SMS hits):
```sh
# the flask listener script is hardcoded to listen on port 8284
ngrok http 8284
```

## Run the MemGPT server

Run the MemGPT server to turn on the agent service:
```sh
memgpt server --debug
```

# Example interaction 

TODO