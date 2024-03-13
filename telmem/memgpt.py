import requests
import json
import asyncio
import logging
from config import MEMGPT_ADMIN_API_KEY
from db import save_user_api_key, save_user_agent_id, get_user_api_key, get_user_agent_id, check_user_exists, save_memgpt_user_id_and_api_key

# Helper function to make asynchronous HTTP requests
async def async_request(method, url, **kwargs):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: requests.request(method, url, **kwargs))
    return response

async def create_memgpt_user(telegram_user_id: int):
    # Check if user already exists in Supabase
    user_exists = await check_user_exists(telegram_user_id)
    if user_exists:
        return "User already exists."

    # Proceed with MemGPT user and agent creation
    response = await async_request('POST', 'http://localhost:8283/admin/users', headers={'Authorization': f'Bearer {MEMGPT_ADMIN_API_KEY}'})
    if response.status_code == 200:
        user_data = response.json()
        user_api_key = user_data['api_key']
        user_memgpt_id = user_data['user_id']  # Corrected from 'id' to 'user_id'
        agent_response = await async_request(
            'POST',
            'http://localhost:8283/api/agents',
            headers={'Authorization': f'Bearer {user_api_key}', 'Content-Type': 'application/json'},
            json={
                "config": {
                    "name": f"AgentForTelegramUser{telegram_user_id}",
                    "preset": "memgpt_chat",
                }
            }
        )
        if agent_response.status_code == 200:
            agent_data = agent_response.json()
            agent_id = agent_data['agent_state']['id']
            # Save API key and agent ID in Supabase
            await save_user_api_key(telegram_user_id, user_api_key)
            await save_user_agent_id(telegram_user_id, agent_id)
            # Save MemGPT user ID and API key in Supabase
            await save_memgpt_user_id_and_api_key(telegram_user_id, user_memgpt_id, user_api_key)
            return "Your MemGPT agent has been created."
        else:
            return "Failed to create MemGPT agent."
    else:
        return "Failed to create MemGPT user."

async def send_message_to_memgpt(telegram_user_id: int, message_text: str):
    user_api_key = await get_user_api_key(telegram_user_id)
    agent_id = await get_user_agent_id(telegram_user_id)
    logging.debug(f"user_api_key: {user_api_key}, agent_id: {agent_id}")  # Add this line for debugging
    if not user_api_key or not agent_id:
        return "No API key or agent found. Please start again."
    response = await async_request(
        'POST',
        'http://localhost:8283/api/agents/message',
        headers={'Authorization': f'Bearer {user_api_key}'},
        json={'agent_id': agent_id, 'message': message_text, 'stream': True, 'role': 'user'}
    )
    if response.status_code == 200:
        memgpt_response = response.json().get('response')
        return memgpt_response
    else:
        return "Failed to send message to MemGPT."
