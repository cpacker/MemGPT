from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram import Update
from config import TELEGRAM_TOKEN
from memgpt import create_memgpt_user, send_message_to_memgpt
import asyncio

async def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    # Create a new MemGPT user with the Telegram chat ID as the identifier
    result = await create_memgpt_user(chat_id)
    # Send the result back to the user
    await context.bot.send_message(chat_id=chat_id, text=result)

async def echo(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    message_text = update.message.text
    # Send the received message to MemGPT and get the response
    response = await send_message_to_memgpt(chat_id, message_text)
    # Send the MemGPT response back to the user
    await context.bot.send_message(chat_id=chat_id, text=response)

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.run_polling()

if __name__ == '__main__':
    main()
