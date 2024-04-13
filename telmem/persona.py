import requests

url = "http://localhost:8283/api/sources"

payload = {
    "name": "PDFs",
    "description": "PDFs will be stored in this source"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": "Bearer ilovellms"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)