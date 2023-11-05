import os

DEFAULT = "cs_phd"


def get_human_text(key=DEFAULT, dir=None):
    if dir is None:
        dir = os.path.join(os.path.dirname(__file__), "examples")
    filename = key if key.endswith(".txt") else f"{key}.txt"
    file_path = os.path.join(dir, filename)

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read().strip()
    else:
        raise FileNotFoundError(f"No file found for key {key}, path={file_path}")
