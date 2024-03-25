import requests
from config import MEMGPT_ADMIN_API_KEY

url = "http://localhost:8283/admin/users"

headers = {"accept": "application/json", 'Authorization': f'Bearer {MEMGPT_ADMIN_API_KEY}'}

response = requests.get(url, headers=headers)

print(response.text)