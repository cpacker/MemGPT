from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Initialize Supabase client with the service role key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

async def get_user_api_key(telegram_user_id: int) -> str:
    data = supabase.table("users").select("api_key").eq("telegram_user_id", telegram_user_id).execute()
    if data.data and len(data.data) > 0:
        return data.data[0]['api_key']
    return None

async def save_user_api_key(telegram_user_id: int, user_api_key: str):
    supabase.table("users").insert({"telegram_user_id": telegram_user_id, "api_key": user_api_key}).execute()

async def save_user_agent_id(telegram_user_id: int, agent_id: str):
    supabase.table("users").update({"agent_id": agent_id}).eq("telegram_user_id", telegram_user_id).execute()

async def get_user_agent_id(telegram_user_id: int) -> str:
    data = supabase.table("users").select("agent_id").eq("telegram_user_id", telegram_user_id).execute()
    if data.data and len(data.data) > 0:
        return data.data[0]['agent_id']
    return None
