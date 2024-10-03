from datetime import datetime

from IPython.display import HTML, display


def pprint(messages):
    """Utility function for pretty-printing the output of client.send_message in notebooks"""

    css_styles = """
    <style>
        .terminal {
            background-color: #002b36;
            color: #839496;
            font-family: 'Courier New', Courier, monospace;
            padding: 10px;
            border-radius: 5px;
        }
        .terminal strong {
            color: #b58900;
        }
        .terminal .function-return {
            color: #2aa198;
        }
        .terminal .internal-monologue {
            color: #d33682;
        }
        .terminal .function-call {
            color: #2aa198;
        }
        .terminal .assistant-message {
            color: #859900;
        }
        .terminal pre {
            color: #839496;
        }
    </style>
    """

    html_content = css_styles + "<div class='terminal'>"
    for message in messages:
        date_formatted = message.date.strftime("%Y-%m-%d %H:%M:%S")

        if hasattr(message, 'function_return'):
            html_content += f"<p><strong>🛠️ [{date_formatted}] Function Return ({message.status}):</strong></p>"
            html_content += f"<p class='function-return'>{message.function_return}</p>"
        elif hasattr(message, 'internal_monologue'):
            html_content += f"<p><strong>💭 [{date_formatted}] Internal Monologue:</strong></p>"
            html_content += f"<p class='internal-monologue'>{message.internal_monologue}</p>"
        elif hasattr(message, 'function_call'):
            html_content += f"<p><strong>🛠️ [{date_formatted}] Function Call:</strong></p>"
            html_content += f"<p class='function-call'>{message.function_call}</p>"
        elif hasattr(message, 'assistant_message'):
            html_content += f"<p><strong>🤖 [{date_formatted}] Assistant Message:</strong></p>"
            html_content += f"<p class='assistant-message'>{message.assistant_message}</p>"
        html_content += "<br>"
    html_content += "</div>"

    display(HTML(html_content))
