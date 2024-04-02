from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from telegram import Update
from memgpt import create_memgpt_user, create_agent, current_agent, delete_agent, change_agent, send_message_to_memgpt, check_user_exists, list_agents, save_memgpt_user_id_and_api_key
import asyncio
import logging
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update: Update, context: CallbackContext):
    try:
        chat_id = update.effective_chat.id
        user_exists = await check_user_exists(chat_id)
        
        if not user_exists:
            # Create a new user in Supabase and MemGPT, and save their details
            creation_response = await create_memgpt_user(chat_id)
            await context.bot.send_message(chat_id=chat_id, text=creation_response)
        else:
            # Inform the user that they already have an account
            await context.bot.send_message(chat_id=chat_id, text="Welcome back! Your account is already set up.")
    except Exception as e:
        print(f"Exception occurred: {e}")
        await context.bot.send_message(chat_id=chat_id, text="An error occurred. Please try again.")

async def echo(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    message_text = update.message.text
    response = await send_message_to_memgpt(chat_id, message_text)
    await context.bot.send_message(chat_id=chat_id, text=response)

# New debug command
async def debug(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Debug: Bot is running.")

async def listagents(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    response = await list_agents(chat_id)
    await context.bot.send_message(chat_id=chat_id, text=response)

async def createagent(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    
    # Check if arguments are provided
    if context.args:
        name = context.args[0]
        response = await create_agent(chat_id, name)
        await context.bot.send_message(chat_id=chat_id, text=response)
    else:
        # If no arguments are provided, send a message asking the user to provide a name
        await context.bot.send_message(chat_id=chat_id, text="Please provide a name for the agent.")

async def currentagent(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    response = await current_agent(chat_id)
    await context.bot.send_message(chat_id=chat_id, text=response)        

# New check user command
async def check_user(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    user_exists = await check_user_exists(chat_id)
    if user_exists:
        await context.bot.send_message(chat_id=chat_id, text="This user is already registered.")
    else:
        await context.bot.send_message(chat_id=chat_id, text="This user is not registered.")

async def changeagent(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    
    # Check if arguments are provided
    if context.args:
        name = context.args[0]
        response = await change_agent(chat_id, name)
        await context.bot.send_message(chat_id=chat_id, text=response)
    else:
        # If no arguments are provided, send a message asking the user to provide a name
        await context.bot.send_message(chat_id=chat_id, text="Please type the name of the agent. Type /listagents.")

async def deleteagent(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    
    # Check if arguments are provided
    if context.args:
        name = context.args[0]
        response = await delete_agent(chat_id, name)
        await context.bot.send_message(chat_id=chat_id, text=response)
    else:
        # If no arguments are provided, send a message asking the user to provide a name
        await context.bot.send_message(chat_id=chat_id, text="Please type the name of the agent. Type /listagents.")


async def help_command(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    help_text = "Available commands:\n"
    help_text += "/start - Creation of user and first agent.\n"
    help_text += "/listagents - List all agents\n"
    help_text += "/currentagent - Display current agent\n"
    help_text += "/changeagent <name> - Change the current agent\n"
    help_text += "/createagent <name> - Create a new agent\n"
    help_text += "/deleteagent <name> - Delete an existing agent\n"
    # help_text += "/debug - Check if bot is running\n"
    help_text += "/check_user - Check if user is registered\n"
    help_text += "/help - Show this help message\n"
    await context.bot.send_message(chat_id=chat_id, text=help_text)

def main():
    logging.basicConfig(level=logging.DEBUG)
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("listagents", listagents))
    application.add_handler(CommandHandler("currentagent", currentagent))
    application.add_handler(CommandHandler("changeagent", changeagent))
    application.add_handler(CommandHandler("createagent", createagent))
    application.add_handler(CommandHandler("deleteagent", deleteagent))
    application.add_handler(CommandHandler("debug", debug))
    application.add_handler(CommandHandler("check_user", check_user))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.run_polling()

if __name__ == '__main__':
    main()
