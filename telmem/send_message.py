import requests
import json

agent_id = 'cc624ea0-d72d-4129-b0e3-860c9a2fa405'
user_api_key = 'sk-1ef3ade1bed22fb514c0571bf08c03871589059d260b05e0'
url = f"http://localhost:8283/api/agents/{agent_id}/messages"

payload = {
    "message": "It is an amazing day today. Weather is great!",
    "stream": True,
    "role": "user"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {user_api_key}"
}

response = requests.post(url, json=payload, headers=headers)

# Extract and print assistant messages
for line in response.text.split('\n'):
    if line.startswith('data:'):
        try:
            data = json.loads(line[len('data:'):])
            if 'assistant_message' in data:
                assistant_message = data['assistant_message']
                print("Assistant Message:", assistant_message)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
