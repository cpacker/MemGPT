from telegram.ext import Application, CommandHandler, MessageHandler, filters
from config import TELEGRAM_TOKEN
from memgpt import create_memgpt_user, send_message_to_memgpt

async def start(update, context):
    # Use create_memgpt_user from memgpt_integration.py

async def echo(update, context):
    # Use send_message_to_memgpt from memgpt_integration.py

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling()

if __name__ == '__main__':
    main()
