---
title: Giving MemGPT more tools
excerpt: Customize your MemGPT agents even further with your own functions
category: 6580daaa48aeca0038fc2297
---

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

### Simple example: giving MemGPT the ability to roll a D20

> ⚠️ Function requirements
>
> The functions you write MUST have proper docstrings and type hints - this is because MemGPT will use these docstrings and types to automatically create a JSON schema that is used in the LLM prompt. Use the docstrings and types annotations from the [example functions](https://github.com/cpacker/MemGPT/blob/main/memgpt/functions/function_sets/base.py) for guidance.

> ⚠️ Function output length
>
> Your custom function should always return a string that is **capped in length**. If your string goes over the specified limit, it will be truncated internally. This is to prevent potential context overflows caused by uncapped string returns (for example, a rogue HTTP request that returns a string larger than the LLM context window).
>
> If you return any type other than `str` (e.g. `dict``) in your custom functions, MemGPT will attempt to cast the result to a string (and truncate the result if it is too long). It is preferable to return strings - think of your function returning a natural language description of the outcome (see the D20 example below).

In this simple example we'll give MemGPT the ability to roll a [D20 die](https://en.wikipedia.org/wiki/D20_System).

First, let's create a python file  `~/.memgpt/functions/d20.py`, and write some code that uses the `random` library to "roll a die":

```python
import random


def roll_d20(self) -> str:
    """
    Simulate the roll of a 20-sided die (d20).

    This function generates a random integer between 1 and 20, inclusive,
    which represents the outcome of a single roll of a d20.

    Returns:
        int: A random integer between 1 and 20, representing the die roll.

    Example:
        >>> roll_d20()
        15  # This is an example output and may vary each time the function is called.
    """
    dice_role_outcome = random.randint(1, 20)
    output_string = f"You rolled a {dice_role_outcome}"
    return output_string
```

Notice how we used [type hints](https://docs.python.org/3/library/typing.html) and [docstrings](https://peps.python.org/pep-0257/#multi-line-docstrings) to describe how the function works. **These are required**, if you do not include them MemGPT will not be able to "link" to your function. This is because MemGPT needs a JSON schema description of how your function works, which we automatically generate for you using the type hints and docstring (which you write yourself).

Next, we'll create a custom preset that includes this new `roll_d20` function. Let's create a YAML file `~/.memgpt/presets/memgpt_d20.yaml`:

```yaml
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
  # roll a d20
  - "roll_d20"
```

Now, let's test that we can create a MemGPT agent that has access to this `roll_d20` function.

1. Run `memgpt configure` and select `memgpt_d20` as the default preset
2. Run `memgpt run` and create a new agent
3. Ask the agent to roll a d20, and make sure it runs the function

<img width="960" alt="image" src="https://github.com/cpacker/MemGPT/assets/8505980/03e78509-3489-4ff6-a5bd-6619aa38af85">

As we can see, MemGPT now has access to the `roll_d20` function! `roll_d20` is a very simple example, but custom functions are a very powerful tool: you can basically give MemGPT access to any arbitrary python code you want! You just have to write the python code + docstrings, then link it to MemGPT via a preset.

### Advanced example: giving MemGPT the ability to use the Jira API

_Example taken from [this pull request](https://github.com/cpacker/MemGPT/pull/282) by @cevatkerim_

As an example, if you wanted to give MemGPT the ability to make calls to Jira Cloud, you would write the function in Python (you would save this python file inside `~/.memgpt/functions/jira_cloud.py`):

```python
import os

from jira import JIRA
from jira.exceptions import JIRAError


def get_jira(self, issue_key: str) -> dict:
    """
    Makes a request to user's JIRA instance with the jira issue id that is provided and returns the issue details

    Args:
        issue_key (str): the issue key (MAIN-1 for example).

    Returns:
        dict: The response from the JIRA request.
    """
    if self.jira is None:
        server = os.getenv("JIRA_SERVER")
        username = os.getenv("JIRA_USER")
        password = os.getenv("JIRA_KEY")
        self.jira = JIRA({"server": server}, basic_auth=(username, password))
    try:
        issue = self.jira.issue(issue_key)
        return {
            "issue": {
                "key": issue.key,
                "summary": issue.fields.summary,
                "description": issue.fields.description,
                "created": issue.fields.created,
                "assignee": issue.fields.creator.displayName,
                "status": issue.fields.status.name,
                "status_category": issue.fields.status.statusCategory.name,
            }
        }
    except JIRAError as e:
        print(f"Error: {e.text}")
        return {"error": str(e.text)}


def run_jql(self, jql: str) -> dict:
    """
    Makes a request to user's JIRA instance with the jql that is provided and returns the issues

    Args:
        jql (str): the JQL.

    Returns:
        dict: The response from the JIRA request.
    """
    if self.jira is None:
        server = os.getenv("JIRA_SERVER")
        username = os.getenv("JIRA_USER")
        password = os.getenv("JIRA_KEY")
        self.jira = JIRA({"server": server}, basic_auth=(username, password))
    try:
        issues = self.jira.search_issues(jql)
        return {"issues": [issue.key for issue in issues]}
    except JIRAError as e:
        print(f"Error: {e.text}")
        return {"error": str(e.text)}
```

Now we need to create a new preset file, let's create one called `~/.memgpt/presets/memgpt_jira.yaml`:

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
  # Jira functions that we made inside of `~/.memgpt/functions/jira_cloud.py`
  - "get_jira"
  - "run_jql"
```

Now when we run `memgpt configure`, we should see the option to use `memgpt_jira` as a preset:

```sh
memgpt configure
```

```text
...
? Select default preset: (Use arrow keys)
   memgpt_extras
   memgpt_docs
   memgpt_chat
 » memgpt_jira
```

Now, if we create a new MemGPT agent (with `memgpt run`) using this `memgpt_jira` preset, it will have the ability to call Jira cloud:
![image](https://github.com/cpacker/MemGPT/assets/1452094/618a3ec3-8d0c-46e9-8a2f-2dbfc3ec57ac)
