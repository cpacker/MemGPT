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


def create_step(step: Step) -> str:
    """
    Creates a step for the current task.

    Args:
        step: A step to add to the task plan.

    Returns:
        str: A summary of the updated task plan after deletion
    """
    DUMMY_MESSAGE = "Step created successfully."
    return DUMMY_MESSAGE
