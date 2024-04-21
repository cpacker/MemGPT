import requests

url = "http://localhost:8283/api/sources"

api = "sk-e310d78c3247e978e0011913c66bb97c84cf8ab573778934"

payload = {
    "name": "Test",
    "description": "PDFs will be stored here"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {api}"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)