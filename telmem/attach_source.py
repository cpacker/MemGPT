import requests

url = "http://localhost:8283/api/sources/c9b7eb23-3759-42e7-b388-c3662b23c2b9/attach?agent_id=10c4d376-5d08-4cf6-814e-916e1f22b393"

headers = {
    "accept": "application/json",
    "authorization": "Bearer ilovellms"
}

response = requests.post(url, headers=headers)

print(response.text)