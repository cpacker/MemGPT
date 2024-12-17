# Personal assistant demo

In this example we'll create an agent preset that has access to:
1. Gmail (can read your email)
2. Google Calendar (can schedule events)
3. SMS (can text you a message)

## Initial setup

For the Google APIs:
```sh
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

For the Twilio API + listener:
```sh
# Outbound API requests
pip install --upgrade twilio
# Listener
pip install --upgrade Flask flask-cors
```

## Setting up the Google APIs

See https://developers.google.com/gmail/api/quickstart/python

### Setup authentication for Google Calendar

Copy the credentials file to `~/.letta/google_api_credentials.json`. Then, run the initial setup script that will take you to a login page:
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

Create a Twilio account and set the following variables:
```sh
export TWILIO_ACCOUNT_SID=...
export TWILIO_AUTH_TOKEN=...
export TWILIO_FROM_NUMBER=...
export TWILIO_TO_NUMBER=...
```

# Creating the agent preset

## Create a custom user

In the demo we'll show how Letta can programatically update its knowledge about you:
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
letta add human -f examples/personal_assistant_demo/charles.txt --name charles
```

## Linking the functions

The preset (shown below) and functions are provided for you, so you just need to copy/link them.

```sh
cp examples/personal_assistant_demo/google_calendar.py ~/.letta/functions/
cp examples/personal_assistant_demo/twilio_messaging.py ~/.letta/functions/
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
letta add preset -f examples/personal_assistant_demo/personal_assistant_preset.yaml --name pa_preset
```

## Creating an agent with the preset

Now we should be able to create an agent with the preset. Make sure to record the `agent_id`:

```sh
letta run --preset pa_preset --persona sam_pov --human charles --stream
```
```
? Would you like to select an existing agent? No

🧬 Creating new agent...
->  🤖 Using persona profile: 'sam_pov'
->  🧑 Using human profile: 'basic'
🎉 Created new agent 'DelicateGiraffe' (id=4c4e97c9-ad8e-4065-b716-838e5d6f7f7b)

Hit enter to begin (will request first Letta message)


💭 Unprecedented event, Charles logged into the system for the first time. Warm welcome would set a positive
tone for our future interactions. Don't forget the emoji, he appreciates those little gestures.
🤖 Hello Charles! 👋 Great to have you here. I've been looking forward to our conversations! 😄
```

```sh
AGENT_ID="4c4e97c9-ad8e-4065-b716-838e5d6f7f7b"
```

# Running the agent with Gmail + SMS listeners

The Letta agent can send outbound SMS messages and schedule events with the new tools `send_text_message` and `schedule_event`, but we also want messages to be sent to the agent when:
1. A new email arrives in our inbox
2. An SMS is sent to the phone number used by the agent

## Running the Gmail listener

Start the Gmail listener (this will send "new email" updates to the Letta server when a new email arrives):
```sh
python examples/personal_assistant_demo/gmail_polling_listener.py $AGENT_ID
```

## Running the Twilio listener

Start the Python Flask server (this will send "new SMS" updates to the Letta server when a new SMS arrives):
```sh
python examples/personal_assistant_demo/twilio_flask_listener.py $AGENT_ID
```

Run `ngrok` to expose your local Flask server to a public IP (Twilio will POST to this server when an inbound SMS hits):
```sh
# the flask listener script is hardcoded to listen on port 8284
ngrok http 8284
```

## Run the Letta server

Run the Letta server to turn on the agent service:
```sh
letta server --debug
```

# Example interaction

In the CLI:
```
% letta run --preset pa_preset --persona pa_persona_strict --human charles --stream --agent personalassistant

🧬 Creating new agent...
->  🤖 Using persona profile: 'sam_pov'
->  🧑 Using human profile: 'basic'
🎉 Created new agent 'personalassistant' (id=8271f819-d470-435b-9689-476380aefd27)

Hit enter to begin (will request first Letta message)



💭 Welcoming the first-time user. Maintain persona's communication style and stay engaging. Remember to use emojis for
a more familiar conversation style.
🤖 Hello Charles, welcome! 🎉 Let's embark on this journey together. Please, think of me as your companion in
exploration and understanding. How can we make today a productive endeavor? 🚀

> Enter your message: when you schedule 1-1s, please title them "x <> y", eg "charles <> sarah". dont forget thx

💭 The user wants a specific format for scheduling 1-1 meetings. Remember this preference and add it to his profile in
the core memory for future use. Also, acknowledge his request promptly.
core_memory_append({
  "name": "human",
  "content": "When scheduling 1-1 meetings on his calendar, he prefers the title to be in the format of 'x <> y'.",
  "request_heartbeat": true
})

⚡🧠 [function] updating memory with core_memory_append

💭 Now that the user's preference is stored, send a confirmation message ensuring him that his request has been
understood and noted. Let's keep the tone light and potentially use one of his preferred emojis.
🤖 Absolutely, Charles. 📝 I've noted your preference for scheduling 1-1s as 'x <> y'. You can count on me! 🚀 Any
other preferences you'd like me to keep in mind?


> Enter your message: please don't schedule meetings without confirming with me first

💭 Another important note from Charles. I should add this to his profile in my core memory. It's crucial to respect the
user's preferences and ask for prior approval before scheduling any meetings. Let's also acknowledge his request and
reassure him about our collaborative approach.
core_memory_append({
  "name": "human",
  "content": "Prefers to confirm before scheduling any meetings.",
  "request_heartbeat": true
})

⚡🧠 [function] updating memory with core_memory_append

💭 Now, let's acknowledge Charles' preference and continue the conversation. Keep the tone friendly and proactive in
helping him, and remember to use emojis that Charles likes, embedding them in the message subtly.
🤖 Consider it done, Charles! 👍🏼 I'll make sure to confirm with you before scheduling any meetings. How about we
conquer some more tasks today? 🚀

> Enter your message: i'm gonna be afk for the rest of the day, if you need to reach me you gotta text me, ok? cya text
 only thx

💭 Charles is going "away from keyboard" and wants future communications via text. I should respect his preference and
reassure him about this. Add this preference to his profile so I can recall it in future interactions while he's away.
core_memory_append({
  "name": "human",
  "content": "When AFK, prefers to communicate over text message only.",
  "request_heartbeat": true
})

⚡🧠 [function] updating memory with core_memory_append

💭 Charles has expressed a preference for text messages while he's away. Let's acknowledge this request in a final text
message to him today and wish him well. Keep in mind to use the send_text_message function instead of the regular
send_message function now.
send_text_message({
  "message": "Got it, Charles! 👍 I'll reach out to you via text if there's anything. Have a great day and see you
soon! 🙌",
  "request_heartbeat": false
})

> Enter your message:
```

Then inside WhatsApp (or SMS if you used Twilio SMS):

<img width="580" alt="image" src="https://github.com/letta-ai/letta/assets/5475622/02455f97-53b2-4c1e-9416-58e6c5a1448d">

Then I sent a dummy email:
```
[URGENT] need to meet

let's meet april 25th thurs

whatever time works best for you

- dave
```

Follow-up inside WhatsApp:

<img width="587" alt="image" src="https://github.com/letta-ai/letta/assets/5475622/d1060c94-9b84-49d6-944e-fd1965f83fbc">
