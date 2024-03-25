import requests

agent_id = 'cc624ea0-d72d-4129-b0e3-860c9a2fa405'
url = f"http://localhost:8283/api/agents/{agent_id}/messages"

payload = {
    "message": "hi",
    "stream": True,
    "role": "user"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer ilovellms"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an error for non-200 status codes

    response_data = response.json()
    if 'messages' in response_data:
        assistant_message = next((msg.get('assistant_message') for msg in response_data['messages'] if 'assistant_message' in msg), None)
        if assistant_message:
            print(assistant_message)
        else:
            print("No 'assistant_message' key found in the MemGPT API response.")
            print("Failed to get answer.")
    else:
        print("No 'messages' key found in the MemGPT API response.")
        print("Failed to get answer.")

except requests.exceptions.RequestException as e:
    print("Error making request to the MemGPT API:", e)
except ValueError as e:
    print("Error decoding JSON from the MemGPT API:", e)
