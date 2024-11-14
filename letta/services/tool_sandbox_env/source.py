import random


def roll_d20() -> str:
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


print(roll_d20())
