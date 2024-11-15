def cowsay() -> str:
    """
    Simple function that uses the cowsay package to print out the secret word env variable.

    Returns:
        str: The cowsay ASCII art.
    """
    import os

    import cowsay

    cowsay.cow(os.getenv("secret_word"))


print(cowsay())
