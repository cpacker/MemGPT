import os

from memgpt.constants import MEMGPT_DIR


def get_system_text(key):
    filename = f"{key}.txt"
    file_path = os.path.join(os.path.dirname(__file__), "system", filename)

    # first look in prompts/system/*.txt
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return file.read().strip()
    else:
        # try looking in ~/.memgpt/system_prompts/*.txt
        user_system_prompts_dir = os.path.join(MEMGPT_DIR, "system_prompts")
        # create directory if it doesn't exist
        if not os.path.exists(user_system_prompts_dir):
            os.makedirs(user_system_prompts_dir)
        # look inside for a matching system prompt
        file_path = os.path.join(user_system_prompts_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                return file.read().strip()
        else:
            raise FileNotFoundError(f"No file found for key {key}, path={file_path}")
