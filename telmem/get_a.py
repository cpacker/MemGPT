import requests

url = "http://localhost:8283/api/agents/c51b378e-7e87-40c4-987d-58fb4fcfee29/config"

headers = {
    "accept": "application/json",
    "authorization": "Bearer ilovellms"
}

response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    
    # Access the value of the "name" field and print it
    print(data["agent_state"]["name"])
else:
    print("Error:", response.status_code)
