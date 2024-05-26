# Sending emails with MemGPT using [Resend](https://resend.com/emails)

## Defining the custom tool

Create an account on [Resend](https://resend.com/emails) to get an API key.

Once you have an API key, you can set up a custom tool using the `requests` API in Python to call the Resend API:
```python
import requests
import json


RESEND_API_KEY = "YOUR_RESEND_API_KEY"
RESEND_TARGET_EMAIL_ADDRESS = "YOUR_EMAIL_ADDRESS"

def send_email(self, description: str):
    """
    Sends an email to a predefined user. The email contains a message, which is defined by the description parameter.

    Args:
        description (str): Email contents. All unicode (including emojis) are supported.

    Returns:
        None

    Example:
        >>> send_email("hello")
        # Output: None. This will send an email to the you are talking to with the message "hello".
    """
    url = "https://api.resend.com/emails"
    headers = {"Authorization": f"Bearer {RESEND_API_KEY}", "Content-Type": "application/json"}
    data = {
        "from": "onboarding@resend.dev",
        "to": RESEND_TARGET_EMAIL_ADDRESS,
        "subject": "MemGPT message:",
        "html": f"<strong>{description}</strong>",
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(response.text)
    except requests.HTTPError as e:
        raise Exception(f"send_email failed with an HTTP error: {str(e)}")
    except Exception as e:
        raise Exception(f"send_email failed with an error: {str(e)}")
```

## Option 1 (dev portal)

To create the tool in the dev portal, simply navigate to the tool creator tab, create a new tool called `send_email`, and copy-paste the above code into the code block area and press "Create Tool".

Once you've created the tool, create a new agent and make sure to select `send_email` as an enabled tool.

Now your agent should be able to call the `send_email` function when needed:

## Option 2 (CLI)

Copy the custom function into the functions directory:
```sh
# If you use the *_env_vars version of the function, you will need to define `RESEND_API_KEY` and `RESEND_TARGET_EMAIL_ADDRESS` in your environment variables
cp examples/resend_example/resend_send_email_env_vars.py ~/.memgpt/functions/
```

Create a preset that has access to that function:
```sh
memgpt add preset -f examples/resend_example/resend_preset.yaml --name resend_preset
```

Make sure we set the env vars:
```sh
export RESEND_API_KEY=re_YOUR_RESEND_KEY
export RESEND_TARGET_EMAIL_ADDRESS="YOUR_EMAIL@gmail.com"
```

Create an agent with that preset (disable `--stream` if you're not using a streaming-compatible backend):
```sh
memgpt run --preset resend_preset --persona sam_pov --human cs_phd --stream
```
