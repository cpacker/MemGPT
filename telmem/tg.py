from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import requests
from supabase import create_client, Client
import json

# Supabase setup
url: str = "https://hglatxkickrrkixkrkgk.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhnbGF0eGtpY2tycmtpeGtya2drIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDk1MTQ0NjMsImV4cCI6MjAyNTA5MDQ2M30.I1kU52TOAa3ELOYPBf9fuaysrdEHvM_rH9wt4F7z7Z4"
supabase: Client = create_client(url, key)

TELEGRAM_TOKEN = '6358193342:AAH7XXW2mYfdl4uWlEFBKvr0fyf4D1Aes0Y'
MEMGPT_ADMIN_API_KEY = 'ilovellms'

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

async def start(update: Update, context):
    telegram_user_id = update.message.from_user.id
    user_api_key = await get_user_api_key(telegram_user_id)
    if not user_api_key:
        response = requests.post('https://memgpt.readme.io/admin/users', headers={'Authorization': f'Bearer {MEMGPT_ADMIN_API_KEY}'})
        print(f"Create user response: {response.json()}")  # Debugging line
        if response.status_code == 200:
            user_data = response.json()
            user_api_key = user_data['api_key']
            # Store this API key with the telegram_user_id for future use
            await save_user_api_key(telegram_user_id, user_api_key)
            # New logic to create an agent for the user
            agent_response = requests.post(
                'https://memgpt.readme.io/api/agents',
                headers={'Authorization': f'Bearer {user_api_key}', 'Content-Type': 'application/json'},
                data=json.dumps({
                    "config": {
                        "name": f"AgentForTelegramUser{telegram_user_id}",
                        "preset": "memgpt_chat",  # Assuming a chat preset, adjust as needed
                        # Add other agent configuration as required
                    }
                })
            )
            if agent_response.status_code == 200:
                agent_data = agent_response.json()
                agent_id = agent_data['agent_state']['id']
                # Store the agent ID with the telegram_user_id for future use
                await save_user_agent_id(telegram_user_id, agent_id)
                await update.message.reply_text("Your MemGPT agent has been created.")
            else:
                await update.message.reply_text("Failed to create MemGPT agent.")
        else:
            await update.message.reply_text("Failed to create MemGPT user.")

async def echo(update: Update, context):
    # Forward message to MemGPT and send response back to Telegram user
    message_text = update.message.text
    telegram_user_id = update.message.from_user.id
    user_api_key = await get_user_api_key(telegram_user_id)
    agent_id = await get_user_agent_id(telegram_user_id)  # Implement this function
    if not user_api_key or not agent_id:
        await update.message.reply_text("No API key or agent found. Please start again.")
        return
    response = requests.post(
        'https://memgpt.readme.io/api/agents/message',
        headers={'Authorization': f'Bearer {user_api_key}'},
        json={'agent_id': agent_id, 'message': message_text, 'stream': True, 'role': 'user'}
    )
    if response.status_code == 200:
        memgpt_response = response.json().get('response')
        await update.message.reply_text(memgpt_response)
    else:
        await update.message.reply_text("Failed to send message to MemGPT.")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling()

if __name__ == '__main__':
    main()
