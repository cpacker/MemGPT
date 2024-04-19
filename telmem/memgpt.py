import requests
import json
import asyncio
import logging
import random
from db import save_user_api_key, save_user_agent_id, get_user_api_key, get_user_agent_id, get_memgpt_user_id, check_user_exists, save_memgpt_user_id_and_api_key
from archival import string

import os
from dotenv import load_dotenv

load_dotenv()

MEMGPT_ADMIN_API_KEY = os.getenv("MEMGPT_SERVER_PASS")

# Helper function to make asynchronous HTTP requests
async def async_request(method, url, **kwargs):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: requests.request(method, url, **kwargs))
    return response

async def create_memgpt_user(telegram_user_id: int):

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
                    "user_id": f"{user_memgpt_id}",
                    "name": "DefaultAgent",
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
            # Insert archival memory about the project
            await insert_archival(agent_id)
            return "Your MemGPT agent has been created."
        else:
            return "Failed to create MemGPT agent."
    else:
        return "Failed to create MemGPT user."

async def send_message_to_memgpt(telegram_user_id: int, message_text: str):
    user_api_key = await get_user_api_key(telegram_user_id)
    agent_id = await get_user_agent_id(telegram_user_id)
    print(f"user_api_key: {user_api_key}, agent_id: {agent_id}")  # Add this line for debugging
    if not user_api_key or not agent_id:
        return "No API key or agent found. Please start again."
    
    response = await async_request(
        'POST',
        f'http://localhost:8283/api/agents/{agent_id}/messages',
        headers={'Authorization': f'Bearer {user_api_key}'},
        json={'agent_id': agent_id, 'message': message_text, 'stream': True, 'role': 'user'}
    )
    
    if response.status_code == 200:
        # Extract and return assistant message
        assistant_message = None
        for line in response.text.split('\n'):
            if line.startswith('data:'):
                try:
                    data = json.loads(line[len('data:'):])
                    if 'assistant_message' in data:
                        assistant_message = data['assistant_message']
                        break
                except json.JSONDecodeError as e:
                    print("Error parsing JSON:", e)
                    
        if assistant_message:
            return assistant_message
        else:
            return "No assistant message found in response."
    else:
        return "Failed to send message to MemGPT."

async def list_agents(telegram_user_id: int):
    # Check if user already exists in Supabase
    user_exists = await check_user_exists(telegram_user_id)
    if not user_exists:
        return "Create a user first."

    user_api_key = await get_user_api_key(telegram_user_id)

    url = "http://localhost:8283/api/agents"
    headers = {"accept": "application/json", "authorization": f"Bearer {user_api_key}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = json.loads(response.text)
        num_agents = data.get("num_agents", 0)
        agents = data.get("agents", [])
        
        agent_info = f"Num of agents: {num_agents}\n" + "-" * 7 + "\n"
        
        for agent in agents:
            name = agent.get("name", "")
            agent_id = agent.get("id", "")
            persona = agent.get("persona", "")
            created_at = agent.get("created_at", "")
            
            agent_info += f"Agent Name: {name}\n"
            agent_info += f"Agent ID: {agent_id}\n"
            agent_info += f"Persona: {persona}\n"
            agent_info += f"Creation Date: {created_at}\n"
            agent_info += "-------\n"
        
        return agent_info
    else:
        return "Failed to fetch agents data."
    
async def create_agent(telegram_user_id: int, agent_name: str):
    # Check if user already exists in Supabase
    user_exists = await check_user_exists(telegram_user_id)
    if not user_exists:
        return "Create a user first."

    user_api_key = await get_user_api_key(telegram_user_id)
    user_memgpt_id = await get_memgpt_user_id(telegram_user_id)
    agent_response = await async_request(
            'POST',
            'http://localhost:8283/api/agents',
            headers={'Authorization': f'Bearer {user_api_key}', 'Content-Type': 'application/json'},
            json={
                "config": {
                    "user_id": f"{user_memgpt_id}",
                    "name" : f"{agent_name}",
                    "preset": "memgpt_chat",
                }
            }
        )



    if agent_response.status_code == 200:
        agents_info = await list_agents(telegram_user_id)

        agent_id = await name_to_id(agents_info, agent_name)

        await insert_archival(agent_id)
        return "Your MemGPT agent has been created."
    else:
        return "Failed to create MemGPT agent."

async def insert_archival(agent_id: str):
    url = f"http://localhost:8283/api/agents/{agent_id}/archival"
    
    payload = {
     "content": string }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {MEMGPT_ADMIN_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)

async def current_agent(telegram_user_id: int):
    # Check if user already exists in Supabase
    user_exists = await check_user_exists(telegram_user_id)
    if not user_exists:
        return "Create a user first."

    user_api_key = await get_user_api_key(telegram_user_id)

    agent_id = await get_user_agent_id(telegram_user_id)

    url = f"http://localhost:8283/api/agents/{agent_id}/config"
    headers = {"accept": "application/json", "authorization": f"Bearer {user_api_key}"}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = json.loads(response.text)
        
        agent_state = data.get("agent_state", {})
        agent_id = agent_state.get("id", "")
        agent_name = agent_state.get("name", "")
        created_at = agent_state.get("created_at", "")
        preset = agent_state.get("preset", "")

        current_agent_info = f"Your current agent info:\n"
        current_agent_info += f"-----{agent_name}-----\n"
        current_agent_info += f"Agent ID: {agent_id}\n"
        current_agent_info += f"Preset: {preset}\n"
        current_agent_info += f"Creation Date: {created_at}\n"

        return current_agent_info
    else:
        return "Failed to fetch current agent's info."

async def change_agent(telegram_user_id: int, agent_name: str):
    # Check if user already exists in Supabase
    user_exists = await check_user_exists(telegram_user_id)
    if not user_exists:
        return "Create a user first."
    
    # Fetch the list of agents
    agents_info = await list_agents(telegram_user_id)
    
    # Check if agent_name matches any of the agent names
    if agent_name not in agents_info:
        return "Agent not found. Please choose from the available agents."

    agent_id = await name_to_id(agents_info, agent_name)

    await save_user_agent_id(telegram_user_id, agent_id)

    return f"Your agent changed to {agent_name}."

async def delete_agent(telegram_user_id: int, agent_name: str):

    changed_agent = False

    # Check if user already exists in Supabase
    user_exists = await check_user_exists(telegram_user_id)
    if not user_exists:
        return "Create a user first."
    
    # Fetch the list of agents
    agents_info = await list_agents(telegram_user_id)
    
    # Check if agent_name matches any of the agent names
    if agent_name not in agents_info:
        return "Agent not found. Please choose from the available agents."

    agent_id = await name_to_id(agents_info, agent_name)
    user_api_key = await get_user_api_key(telegram_user_id)

    curr_agent_id = await get_user_agent_id(telegram_user_id)

    if(agent_id == curr_agent_id):
        if agents_info.count("Agent Name:") == 1:
            return "Please create another agent first."
        
        # Extract agent names
        agent_names = [line.split("Agent Name: ")[1].strip() for line in agents_info.split("\n") if line.startswith("Agent Name:")]

        # Remove current agent from the list
        agent_names.remove(agent_name)

        # Choose a random agent from the remaining list
        new_agent_name = random.choice(agent_names)

        # Change to the new agent
        await change_agent(telegram_user_id, new_agent_name)

        changed_agent = True


    

    url = f"http://localhost:8283/api/agents/{agent_id}"
    headers = {"accept": "application/json", "authorization": f"Bearer {user_api_key}"}

    response = requests.delete(url, headers=headers)

    if response.status_code == 200:
        if changed_agent:
            return f"Agent {agent_name} successfully deleted and current agent changed to {new_agent_name}."
        return f"Agent {agent_name} successfully deleted."
    else:
        return f"Error occured."

async def name_to_id(agents_info, agent_name):
    # Split agents_info into individual agent records
    agent_records = agents_info.split("-------")
    agent_id = None

    # Search for the agent name and extract its ID
    for record in agent_records:
        if agent_name in record:
            lines = record.split("\n")
            for line in lines:
                if line.startswith("Agent ID:"):
                    agent_id = line.split("Agent ID: ")[1].strip()
                    break
            if agent_id:
                break

    if agent_id is None:
        return "Failed to extract agent ID."
    
    return agent_id
    