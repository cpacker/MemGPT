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

### Example: giving MemGPT the ability to use the Jira API

!!! warning "Function requirements"

    The functions you write MUST have proper docstrings and type hints - this is because MemGPT will use these docstrings and types to automatically create a JSON schema that is used in the LLM prompt. Use the docstrings and types annotations from the [example functions](https://github.com/cpacker/MemGPT/blob/main/memgpt/functions/function_sets/base.py) for guidance.

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
