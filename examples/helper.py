# Add your utilities or helper functions to this file.

import html
import json
import os
import re

from dotenv import find_dotenv, load_dotenv
from IPython.display import HTML, display


# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService
def load_env():
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def nb_print(messages):
    html_output = """
    <style>
        .message-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 20px auto;
            background-color: #1e1e1e;
            border-radius: 8px;
            overflow: hidden;
            color: #d4d4d4;
        }
        .message {
            padding: 10px 15px;
            border-bottom: 1px solid #3a3a3a;
        }
        .message:last-child {
            border-bottom: none;
        }
        .title {
            font-weight: bold;
            margin-bottom: 5px;
            color: #ffffff;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        .content {
            background-color: #2d2d2d;
            border-radius: 4px;
            padding: 5px 10px;
            font-family: 'Consolas', 'Courier New', monospace;
            white-space: pre-wrap;
        }
        .status-line {
            margin-bottom: 5px;
            color: #d4d4d4;
        }
        .function-name { color: #569cd6; }
        .json-key { color: #9cdcfe; }
        .json-string { color: #ce9178; }
        .json-number { color: #b5cea8; }
        .json-boolean { color: #569cd6; }
        .internal-monologue { font-style: italic; }
    </style>
    <div class="message-container">
    """

    for msg in messages:
        content = get_formatted_content(msg)

        # don't print empty function returns
        if msg.message_type == "function_return":
            return_data = json.loads(msg.function_return)
            if "message" in return_data and return_data["message"] == "None":
                continue

        title = msg.message_type.replace("_", " ").upper()
        html_output += f"""
        <div class="message">
            <div class="title">{title}</div>
            {content}
        </div>
        """

    html_output += "</div>"
    display(HTML(html_output))


def get_formatted_content(msg):
    if msg.message_type == "internal_monologue":
        return f'<div class="content"><span class="internal-monologue">{html.escape(msg.internal_monologue)}</span></div>'
    elif msg.message_type == "function_call":
        args = format_json(msg.function_call.arguments)
        return f'<div class="content"><span class="function-name">{html.escape(msg.function_call.name)}</span>({args})</div>'
    elif msg.message_type == "function_return":

        return_value = format_json(msg.function_return)
        # return f'<div class="status-line">Status: {html.escape(msg.status)}</div><div class="content">{return_value}</div>'
        return f'<div class="content">{return_value}</div>'
    elif msg.message_type == "user_message":
        if is_json(msg.message):
            return f'<div class="content">{format_json(msg.message)}</div>'
        else:
            return f'<div class="content">{html.escape(msg.message)}</div>'
    elif msg.message_type in ["assistant_message", "system_message"]:
        return f'<div class="content">{html.escape(msg.message)}</div>'
    else:
        return f'<div class="content">{html.escape(str(msg))}</div>'


def is_json(string):
    try:
        json.loads(string)
        return True
    except ValueError:
        return False


def format_json(json_str):
    try:
        parsed = json.loads(json_str)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        formatted = formatted.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        formatted = formatted.replace("\n", "<br>").replace("  ", "&nbsp;&nbsp;")
        formatted = re.sub(r'(".*?"):', r'<span class="json-key">\1</span>:', formatted)
        formatted = re.sub(r': (".*?")', r': <span class="json-string">\1</span>', formatted)
        formatted = re.sub(r": (\d+)", r': <span class="json-number">\1</span>', formatted)
        formatted = re.sub(r": (true|false)", r': <span class="json-boolean">\1</span>', formatted)
        return formatted
    except json.JSONDecodeError:
        return html.escape(json_str)
