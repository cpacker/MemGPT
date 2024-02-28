from openai import OpenAI
import time

"""
This script provides an example of how you can use OpenAI's python client with a MemGPT server.

Before running this example, make sure you start the OpenAI-compatible REST server with `memgpt server`.
"""


def main():
    client = OpenAI(base_url="http://localhost:8283/v1")

    # create assistant (creates a memgpt preset)
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        model="gpt-4-turbo-preview",
    )

    # create thread (creates a memgpt agent)
    thread = client.beta.threads.create()

    # create a message (appends a message to the memgpt agent)
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
    )

    # create a run (executes the agent on the messages)
    # NOTE: MemGPT does not support polling yet, so run status is always "completed"
    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id, instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    # Store the run ID
    run_id = run.id

    # Retrieve all messages from the thread
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Print all messages from the thread
    for msg in messages.messages:
        role = msg["role"]
        content = msg["content"][0]
        print(f"{role.capitalize()}: {content}")


if __name__ == "__main__":
    main()
