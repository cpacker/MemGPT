from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from memgpt import create_memgpt_user, create_agent, current_agent, delete_agent, change_agent, send_message_to_memgpt, check_user_exists, list_agents
import logging
import os
from dotenv import load_dotenv
import re
from db import save_user_pseudonym, get_user_info, delete_user as db_delete_user
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# At the top of your script, add a dictionary for state management
user_states = {}

async def start(update: Update, context: CallbackContext):
    chat_id = update.message.chat.id
    user_id = update.message.from_user.id
    user_exists = await check_user_exists(user_id)
    
    if not user_exists:
        await create_memgpt_user(user_id)
        await context.bot.send_message(chat_id=chat_id, text="Welcome to ƒxyz Network! Please choose a pseudonym for your interactions within our network. This will be your unique identifier and help maintain your privacy.")
        user_states[user_id] = 'awaiting_pseudonym'
    else:
        await context.bot.send_message(chat_id=chat_id, text="Welcome back! Your account is already set up.")

async def echo(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    chat_id = update.message.chat.id
    message_text = update.message.text

    if user_states.get(user_id) == 'awaiting_pseudonym':
        # Save the pseudonym to Supabase
        success = await save_user_pseudonym(user_id, message_text)
        if success:
            user_states[user_id] = None  # Reset the state
            await context.bot.send_message(chat_id=chat_id, text="Your agent has been created. You can now send messages to your new agent or use /menu to manage your agents.")
        else:
            await context.bot.send_message(chat_id=chat_id, text="There was an error saving your pseudonym. Please try again.")
    elif user_states.get(user_id) == 'awaiting_agent_name':
        response = await create_agent(user_id, message_text)
        response += "\nAgent created successfully! You can change your current agent to start chatting with your new agent."
        await context.bot.send_message(chat_id=chat_id, text=response)
        user_states[user_id] = None
    else:
        # Handle other messages normally
        if update.message.chat.type == "group":
            bot_username = context.bot.username
            if bot_username in update.message.text:
                # Echo the received message back to the sender
                response = await send_message_to_memgpt(user_id, message_text)
                await context.bot.send_message(chat_id=chat_id, text=response)
        else:
            response = await send_message_to_memgpt(user_id, message_text)
            await context.bot.send_message(chat_id=chat_id, text=response)

async def debug(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.message.from_user.id, text="Debug: Bot is running.")

async def user_info(update: Update, context: CallbackContext):
    user_id = update.callback_query.from_user.id
    chat_id = update.callback_query.message.chat.id
    
    # Fetch user information from the database
    user_info = await get_user_info(user_id)
    if user_info:
        message = f"Your information:\n\nPseudonym: {user_info['pseudonym']}\nPreferred Language: {user_info.get('language', 'Not set')}\nMemGPT Agent ID: {user_info.get('agent_id', 'Not set')}"
        await context.bot.send_message(chat_id=chat_id, text=message)
    else:
        await context.bot.send_message(chat_id=chat_id, text="User information not found. Please register first.")

async def delete_user_profile(update: Update, context: CallbackContext):
    chat_id = update.callback_query.message.chat.id
    keyboard = [
        [InlineKeyboardButton("Yes", callback_data='confirm_delete_user')],
        [InlineKeyboardButton("No", callback_data='cancel_delete_user')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=chat_id, text="Are you sure you want to delete your profile?", reply_markup=reply_markup)

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
    help_text += "/menu - Check if user is registered\n"
    help_text += "/help - Show this help message\n"
    help_text += "/userinfo - Check your user information\n"
    help_text += "/deleteuser - Delete your user profile\n"
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
        ],
        [
            InlineKeyboardButton("User Info", callback_data='userinfo'),
            InlineKeyboardButton("Delete User", callback_data='deleteuser')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Please choose an option:", reply_markup=reply_markup)

async def button_click(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()
    callback_data = query.data
    user_id = query.from_user.id
    chat_id = query.message.chat.id

    if callback_data == 'createagent':
        # Set the user's state to 'awaiting_agent_name'
        user_states[user_id] = 'awaiting_agent_name'
        await context.bot.send_message(chat_id=chat_id, text="Please send the name for your new agent.")
    elif callback_data == 'list_agents':
        response = await list_agents(user_id)
        await context.bot.send_message(chat_id=chat_id, text=response)
    elif callback_data == 'current_agent':
        response = await current_agent(user_id)
        await context.bot.send_message(chat_id=chat_id, text=response)
    elif callback_data == 'change_agent_buttons':
        await change_agent_buttons(update, context)
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
    elif callback_data == 'userinfo':
        await user_info(update, context)
    elif callback_data == 'deleteuser':
        await delete_user_profile(update, context)
    elif callback_data == 'confirm_delete_user':
        user_id = update.callback_query.from_user.id
        chat_id = update.callback_query.message.chat.id
        # Correctly call the delete_user function from db.py
        success = await db_delete_user(user_id)
        if success:
            await context.bot.send_message(chat_id=chat_id, text="Your user profile has been successfully deleted.")
        else:
            await context.bot.send_message(chat_id=chat_id, text="Failed to delete user profile. Please try again later.")
    elif callback_data == 'cancel_delete_user':
        chat_id = update.callback_query.message.chat.id
        await context.bot.send_message(chat_id=chat_id, text="Profile deletion cancelled.")

def main():
    logging.basicConfig(level=logging.DEBUG)
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("menu", menu))
    application.add_handler(CommandHandler("createagent", createagent))
    application.add_handler(CommandHandler("userinfo", user_info))
    application.add_handler(CommandHandler("deleteuser", delete_user_profile))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.add_handler(CallbackQueryHandler(callback=button_click))

    application.run_polling()

if __name__ == '__main__':
    main()
