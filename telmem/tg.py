from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram import Update
from config import TELEGRAM_TOKEN
from memgpt import create_memgpt_user, send_message_to_memgpt, check_user_exists
import asyncio

async def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    await context.bot.send_message(chat_id=chat_id, text="Welcome! Let me check if you're already registered.")
    user_exists = await check_user_exists(chat_id)
    if user_exists:
        await context.bot.send_message(chat_id=chat_id, text="You're already registered. How can I assist you today?")
    else:
        result = await create_memgpt_user(chat_id)
        await context.bot.send_message(chat_id=chat_id, text=f"Account created successfully! {result}")

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
