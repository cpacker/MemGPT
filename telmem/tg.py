from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram import Update
from config import TELEGRAM_TOKEN
from memgpt import create_memgpt_user, send_message_to_memgpt, check_user_exists, save_user_api_key
import asyncio

async def start(update: Update, context: CallbackContext):
    try:
        chat_id = update.effective_chat.id
        user_exists = await check_user_exists(chat_id)
        
        if not user_exists:
            # Create a new user in Supabase and MemGPT
            creation_response = await create_memgpt_user(chat_id)
            await context.bot.send_message(chat_id=chat_id, text=creation_response)
            # Proceed to create an agent for the new user here if necessary
        else:
            # Inform the user that they already have an account
            await context.bot.send_message(chat_id=chat_id, text="Welcome back! Your account is already set up.")
            # Proceed with existing user flow here
    except Exception as e:
        print(f"Exception occurred: {e}")

async def echo(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    message_text = update.message.text
    response = await send_message_to_memgpt(chat_id, message_text)
    await context.bot.send_message(chat_id=chat_id, text=response)

# New debug command
async def debug(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Debug: Bot is running.")

# New check user command
async def check_user(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    user_exists = await check_user_exists(chat_id)
    if user_exists:
        await context.bot.send_message(chat_id=chat_id, text="This user is already registered.")
    else:
        await context.bot.send_message(chat_id=chat_id, text="This user is not registered.")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("debug", debug))
    application.add_handler(CommandHandler("check_user", check_user))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling()

if __name__ == '__main__':
    main()
