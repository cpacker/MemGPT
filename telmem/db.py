from supabase import create_client, Client
import os
from dotenv import load_dotenv
import postgrest.exceptions
import asyncio
import logging

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase client with the service role key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

logging.basicConfig(level=logging.INFO)

async def get_user_api_key(telegram_user_id: int) -> str:
    loop = asyncio.get_event_loop()
    data, error = await loop.run_in_executor(None, lambda: supabase.table("users").select("api_key").eq("telegram_user_id", telegram_user_id).execute())
    logging.info(f"Data fetched for API key: {data}, Error: {error}")  # Add this for debugging
    if not error and data and len(data) > 0:
        return data[0]['api_key']
    return None

async def save_user_api_key(telegram_user_id: int, user_api_key: str):
    loop = asyncio.get_event_loop()
    data, error = await loop.run_in_executor(None, lambda: supabase.table("users").upsert({"telegram_user_id": telegram_user_id, "api_key": user_api_key}).execute())
    if error and isinstance(error, tuple) and error[0] != 'count':
        raise Exception(f"Failed to save or update user API key: {error}")

async def save_user_agent_id(telegram_user_id: int, agent_id: str):
    try:
        loop = asyncio.get_event_loop()
        data, error = await loop.run_in_executor(None, lambda: supabase.table("users").upsert({"telegram_user_id": telegram_user_id, "agent_id": agent_id}).execute())
        if error and isinstance(error, tuple) and error[0] != 'count':
            raise Exception(f"Failed to save or update agent ID: {error}")
    except postgrest.exceptions.APIError as e:
        if e.message == 'duplicate key value violates unique constraint "users_telegram_user_id_key"':
            print(f"Duplicate key error for telegram_user_id: {telegram_user_id}. Record might already exist.")
        else:
            raise

async def get_user_agent_id(telegram_user_id: int) -> str:
    loop = asyncio.get_event_loop()
    data, error = await loop.run_in_executor(None, lambda: supabase.table("users").select("agent_id").eq("telegram_user_id", telegram_user_id).execute())
    if not error and data and len(data) > 0:
        return data[0]['agent_id']
    return None

async def check_user_exists(telegram_user_id: int) -> bool:
    try:
        loop = asyncio.get_event_loop()
        data, error = await loop.run_in_executor(None, lambda: supabase.table("users").select("id").eq("telegram_user_id", telegram_user_id).execute())
        if error:
            logging.error(f"Error checking user exists: {error}")
            return False
        return bool(data and len(data) > 0)
    except Exception as e:
        logging.exception("Unexpected error checking if user exists", exc_info=e)
        return False

async def save_memgpt_user_id_and_api_key(telegram_user_id: int, memgpt_user_id: str, user_api_key: str):
    loop = asyncio.get_event_loop()
    data, error = await loop.run_in_executor(None, lambda: supabase.table("users").update({"memgpt_user_id": memgpt_user_id, "api_key": user_api_key}).eq("telegram_user_id", telegram_user_id).execute())
    if error and isinstance(error, tuple) and error[0] != 'count':
        raise Exception(f"Failed to save MemGPT user ID and API key: {error}")
    return data
