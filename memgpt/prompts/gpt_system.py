import os
from memgpt.constants import MEMGPT_DIR


def get_system_text(key):
    filename = f"{key}.txt"
    file_path = os.path.join(os.path.dirname(__file__), "system", filename)
    user_path = os.path.join(MEMGPT_DIR, "prompts", filename)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read().strip()
    elif os.path.exists(user_path):
        with open(user_path, "r") as file:
            return file.read().strip()
    else:
        raise FileNotFoundError(f"No file found for key {key}, path={file_path} and path={user_path}")
