from pydantic import BaseModel, Field


class Step(BaseModel):
    name: str = Field(
        ...,
        description="Name of the step.",
    )
    key: str = Field(
        ...,
        description="Unique identifier for the step.",
    )
    description: str = Field(
        ...,
        description="An exhaustic description of what this step is trying to achieve and accomplish.",
    )


def create_task_plan(steps: list[Step]) -> str:
    """
    Creates a task plan for the current task.
    It takes in a list of steps, and updates the task with the new steps provided.
    If there are any current steps, they will be overwritten.
    Each step in the list should have the following format:
    {
        "name": <string> -- Name of the step.
        "key": <string> -- Unique identifier for the step.
        "description": <string> -- An exhaustic description of what this step is trying to achieve and accomplish.
    }

    Args:
        steps: List of steps to add to the task plan.

    Returns:
        str: A summary of the updated task plan after deletion
    """
    DUMMY_MESSAGE = "Task plan created successfully."
    return DUMMY_MESSAGE
