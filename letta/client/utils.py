import re
from datetime import datetime
from typing import Optional

from IPython.display import HTML, display
from sqlalchemy.testing.plugin.plugin_base import warnings

from letta.local_llm.constants import (
    ASSISTANT_MESSAGE_CLI_SYMBOL,
    INNER_THOUGHTS_CLI_SYMBOL,
)


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
        date_str = message["date"]
        date_formatted = datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")

        if "function_return" in message:
            return_string = message["function_return"]
            return_status = message["status"]
            html_content += f"<p><strong>🛠️ [{date_formatted}] Function Return ({return_status}):</strong></p>"
            html_content += f"<p class='function-return'>{return_string}</p>"
        elif "internal_monologue" in message:
            html_content += f"<p><strong>{INNER_THOUGHTS_CLI_SYMBOL} [{date_formatted}] Internal Monologue:</strong></p>"
            html_content += f"<p class='internal-monologue'>{message['internal_monologue']}</p>"
        elif "function_call" in message:
            html_content += f"<p><strong>🛠️ [[{date_formatted}] Function Call:</strong></p>"
            html_content += f"<p class='function-call'>{message['function_call']}</p>"
        elif "assistant_message" in message:
            html_content += f"<p><strong>{ASSISTANT_MESSAGE_CLI_SYMBOL} [{date_formatted}] Assistant Message:</strong></p>"
            html_content += f"<p class='assistant-message'>{message['assistant_message']}</p>"
        html_content += "<br>"
    html_content += "</div>"

    display(HTML(html_content))


def derive_function_name_regex(function_string: str) -> Optional[str]:
    # Regular expression to match the function name
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", function_string)

    if match:
        function_name = match.group(1)
        return function_name
    else:
        warnings.warn("No function name found.")
        return None
