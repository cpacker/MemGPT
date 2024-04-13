from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from memgpt import create_memgpt_user, create_agent, current_agent, delete_agent, change_agent, send_message_to_memgpt, check_user_exists, list_agents
import logging
import os
from dotenv import load_dotenv
import re

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

async def start(update: Update, context: CallbackContext):
    try:
        user_id = update.message.from_user.id
        user_exists = await check_user_exists(user_id)
        chat_id = update.message.chat.id
        
        if not user_exists:
            # Create a new user in Supabase and MemGPT, and save their details
            creation_response = await create_memgpt_user(user_id)
            chat_id = update.message.chat.id
            await context.bot.send_message(chat_id=chat_id, text=creation_response)
        else:
            # Inform the user that they already have an account
            await context.bot.send_message(chat_id=chat_id, text="Welcome back! Your account is already set up.")
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        await context.bot.send_message(chat_id=chat_id, text="An error occurred. Please try again.")

async def echo(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    message_text = update.message.text
    if update.message.chat.type == "group":
        bot_username = context.bot.username
        if bot_username in update.message.text:
            # Echo the received message back to the sender
            response = await send_message_to_memgpt(user_id, message_text)
            chat_id = update.message.chat.id
            await context.bot.send_message(chat_id=chat_id, text=response)
    else:
        response = await send_message_to_memgpt(user_id, message_text)
        chat_id = update.message.chat.id
        await context.bot.send_message(chat_id=chat_id, text=response)

async def debug(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.message.from_user.id, text="Debug: Bot is running.")


async def createagent(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    chat_id = update.message.chat.id
    
    # Check if arguments are provided
    if context.args:
        name = context.args[0]
        response = await create_agent(user_id, name)
        await context.bot.send_message(chat_id=chat_id, text=response)
    else:
        # If no arguments are provided, send a message asking the user to provide a name
        await context.bot.send_message(chat_id=chat_id, text="Please provide a name for the agent.")


# Generic function to generate inline keyboard markup for a list of agents
async def generate_agent_buttons(update: Update, context: CallbackContext, agents_info: str, prefix: str):
    try:
        if agents_info.startswith("Num of agents"):
            agent_names = re.findall(r'Agent Name: (\S+)', agents_info)
            if agent_names:
                keyboard = [
                    [InlineKeyboardButton(agent, callback_data=f'{prefix}_{agent}') for agent in agent_names]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_message(chat_id=update.callback_query.message.chat.id, text="Choose an agent:", reply_markup=reply_markup)
            else:
                await context.bot.send_message(chat_id=update.callback_query.message.chat.id, text="No agents available.")
        else:
            await context.bot.send_message(chat_id=update.callback_query.message.chat.id, text=agents_info)
    except Exception as e:
        await context.bot.send_message(chat_id=update.callback_query.message.chat.id, text=f"Error: {e}")

async def delete_agent_buttons(update: Update, context: CallbackContext):
    user_id = update.callback_query.from_user.id
    agents_info = await list_agents(user_id)
    await generate_agent_buttons(update, context, agents_info, 'delete')

async def change_agent_buttons(update: Update, context: CallbackContext):
    user_id = update.callback_query.from_user.id
    agents_info = await list_agents(user_id)
    await generate_agent_buttons(update, context, agents_info, 'change')

async def help_command(update: Update, context: CallbackContext):
    chat_id = update.message.chat.id
    help_text = "Available commands:\n"
    help_text += "/start - Creation of user and first agent.\n"
    # help_text += "/debug - Check if bot is running\n"
    help_text += "/menu - Check if user is registered\n"
    help_text += "/help - Show this help message\n"
    await context.bot.send_message(chat_id=chat_id, text=help_text)

async def menu(update: Update, context: CallbackContext):
    # Create a menu with inline buttons
    keyboard = [
        [
            InlineKeyboardButton("List Agents", callback_data='list_agents'),
            InlineKeyboardButton("Current Agent", callback_data='current_agent')
        ],
        [
            InlineKeyboardButton("Change Agent", callback_data='change_agent_buttons'),
            InlineKeyboardButton("Create Agent", callback_data='createagent')
        ],
        [
            InlineKeyboardButton("Delete Agent", callback_data='delete_agent_buttons'),
            InlineKeyboardButton("Check User", callback_data='check_user')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=update.message.chat_id, text="Please select an option:", reply_markup=reply_markup)

async def button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    
    await query.answer()
    
    callback_data = query.data
    user_id = query.from_user.id
    chat_id = query.message.chat.id
    if callback_data == 'list_agents':
        response = await list_agents(user_id)
        await context.bot.send_message(chat_id=chat_id, text=response)
    elif callback_data == 'current_agent':
        response = await current_agent(user_id)
        await context.bot.send_message(chat_id=chat_id, text=response)
    elif callback_data == 'change_agent_buttons':
        await change_agent_buttons(update, context)
    elif callback_data == 'createagent':
        await createagent(update, context)
    elif callback_data == 'delete_agent_buttons':
        await delete_agent_buttons(update, context)
    elif callback_data == 'check_user':
        response = await check_user_exists(user_id)
        await context.bot.send_message(chat_id=chat_id, text=response)
    elif callback_data.startswith('change_'):
        agent_name = callback_data.split('_')[1]
        response = await change_agent(user_id, agent_name)
        await context.bot.send_message(chat_id=chat_id, text=response)
    elif callback_data.startswith('delete_'):
        agent_name = callback_data.split('_')[1]
        response = await delete_agent(user_id, agent_name)
        await context.bot.send_message(chat_id=chat_id, text=response)

def main():
    logging.basicConfig(level=logging.DEBUG)
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("menu", menu))
    application.add_handler(CommandHandler("createagent", createagent))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.add_handler(CallbackQueryHandler(callback=button_click))

    application.run_polling()

if __name__ == '__main__':
    main()
