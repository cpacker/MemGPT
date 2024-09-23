from logging import getLogger

from openai import APIConnectionError, OpenAI

logger = getLogger(__name__)


def test_openai_assistant():
    client = OpenAI(base_url="http://127.0.0.1:8080/v1")
    # create assistant
    try:
        assistant = client.beta.assistants.create(
            name="Math Tutor",
            instructions="You are a personal math tutor. Write and run code to answer math questions.",
            # tools=[{"type": "code_interpreter"}],
            model="gpt-4-turbo-preview",
        )
    except APIConnectionError as e:
        logger.error("Connection issue with localhost openai stub: %s", e)
        return
    # create thread
    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id, assistant_id=assistant.id, instructions="Please address the user as Jane Doe. The user has a premium account."
    )

    # run = client.beta.threads.runs.create(
    #  thread_id=thread.id,
    #  assistant_id=assistant.id,
    #  model="gpt-4-turbo-preview",
    #  instructions="New instructions that override the Assistant instructions",
    #  tools=[{"type": "code_interpreter"}, {"type": "retrieval"}]
    # )

    # Store the run ID
    run_id = run.id
    print(run_id)

    # NOTE: Letta does not support polling yet, so run status is always "completed"
    # Retrieve all messages from the thread
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Print all messages from the thread
    for msg in messages.messages:
        role = msg["role"]
        content = msg["content"][0]
        print(f"{role.capitalize()}: {content}")

    # TODO: add once polling works
    ## Polling for the run status
    # while True:
    #    # Retrieve the run status
    #    run_status = client.beta.threads.runs.retrieve(
    #        thread_id=thread.id,
    #        run_id=run_id
    #    )

    #    # Check and print the step details
    #    run_steps = client.beta.threads.runs.steps.list(
    #        thread_id=thread.id,
    #        run_id=run_id
    #    )
    #    for step in run_steps.data:
    #        if step.type == 'tool_calls':
    #            print(f"Tool {step.type} invoked.")

    #        # If step involves code execution, print the code
    #        if step.type == 'code_interpreter':
    #            print(f"Python Code Executed: {step.step_details['code_interpreter']['input']}")

    #    if run_status.status == 'completed':
    #        # Retrieve all messages from the thread
    #        messages = client.beta.threads.messages.list(
    #            thread_id=thread.id
    #        )

    #        # Print all messages from the thread
    #        for msg in messages.data:
    #            role = msg.role
    #            content = msg.content[0].text.value
    #            print(f"{role.capitalize()}: {content}")
    #        break  # Exit the polling loop since the run is complete
    #    elif run_status.status in ['queued', 'in_progress']:
    #        print(f'{run_status.status.capitalize()}... Please wait.')
    #        time.sleep(1.5)  # Wait before checking again
    #    else:
    #        print(f"Run status: {run_status.status}")
    #        break  # Exit the polling loop if the status is neither 'in_progress' nor 'completed'
