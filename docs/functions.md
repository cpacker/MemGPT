## Giving MemGPT access to additional tools / functions

If you would like to give MemGPT the ability to call new tools or functions, you can write a Python `.py` file with the functions you want to add, and place it inside of `~/.memgpt/functions`. You can see the example function sets provided [here](https://github.com/cpacker/MemGPT/tree/main/memgpt/functions/function_sets).

As an example, we provide a preset called [`memgpt_extras`](https://github.com/cpacker/MemGPT/blob/main/memgpt/presets/examples/memgpt_extras.yaml) that includes additional functions to read and write from text files, as well as make HTTP requests:
```yaml
# this preset uses the same "memgpt_chat" system prompt, but has more functions enabled
system_prompt: "memgpt_chat"
functions:
  - "send_message"
  - "pause_heartbeats"
  - "core_memory_append"
  - "core_memory_replace"
  - "conversation_search"
  - "conversation_search_date"
  - "archival_memory_insert"
  - "archival_memory_search"
  # extras for read/write to files
  - "read_from_text_file"
  - "append_to_text_file"
  # internet access
  - "http_request"
```

### Writing your own functions and connecting them to MemGPT

There are three steps to adding more MemGPT functions:

1. Write the functions themselves in Python
2. (Optional) Create a new system prompt that instructs MemGPT how to use these functions
3. Create a new preset that imports these functions (and optionally uses the new system prompt)

### Step 1: Writing the functions

!!! warning "Function requirements"

    The functions you write MUST have proper docstrings and type hints - this is because MemGPT will use these docstrings and types to automatically create a JSON schema that is used in the LLM prompt. Use the docstrings and types annotations from the [example functions](https://github.com/cpacker/MemGPT/blob/main/memgpt/functions/function_sets/base.py) for guidance.

As an example, if you wanted to give MemGPT the ability to make HTTP requests, you would write the function in Python (you would save this python file inside `~/.memgpt/functions/your_new_functions.py`):
```python
import json
import requests

def http_request(self, method: str, url: str, payload_json: Optional[str] = None):
    """
    Generates an HTTP request and returns the response.

    Args:
        method (str): The HTTP method (e.g., 'GET', 'POST').
        url (str): The URL for the request.
        payload_json (Optional[str]): A JSON string representing the request payload.

    Returns:
        dict: The response from the HTTP request.
    """
    try:
        headers = {"Content-Type": "application/json"}

        # For GET requests, ignore the payload
        if method.upper() == "GET":
            print(f"[HTTP] launching GET request to {url}")
            response = requests.get(url, headers=headers)
        else:
            # Validate and convert the payload for other types of requests
            if payload_json:
                payload = json.loads(payload_json)
            else:
                payload = {}
            print(f"[HTTP] launching {method} request to {url}, payload=\n{json.dumps(payload, indent=2)}")
            response = requests.request(method, url, json=payload, headers=headers)

        return {"status_code": response.status_code, "headers": dict(response.headers), "body": response.text}
    except Exception as e:
        return {"error": str(e)}
```

### Step 3: Create a new preset file

Now we need to create a new preset file, let's create one called `~/.memgpt/presets/memgpt_http.yaml`:
```yaml
# if we had created a new system prompt, we would replace "memgpt_chat" with the new prompt filename (no .txt)
system_prompt: "memgpt_chat"
functions:
  - "send_message"
  - "pause_heartbeats"
  - "core_memory_append"
  - "core_memory_replace"
  - "conversation_search"
  - "conversation_search_date"
  - "archival_memory_insert"
  - "archival_memory_search"
  # internet access
  - "http_request"
```

Now when we run `memgpt configure`, we should see the option to use `memgpt_http` as a preset.