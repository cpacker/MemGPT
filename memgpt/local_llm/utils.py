import os


class DotDict(dict):
    """Allow dot access on properties similar to OpenAI response object"""

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value


def load_grammar_file(grammar):
    # Set grammar
    grammar_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grammars", f"{grammar}.gbnf")

    # Check if the file exists
    if not os.path.isfile(grammar_file):
        # If the file doesn't exist, raise a FileNotFoundError
        raise FileNotFoundError(f"The grammar file {grammar_file} does not exist.")

    with open(grammar_file, "r") as file:
        grammar_str = file.read()

    return grammar_str
