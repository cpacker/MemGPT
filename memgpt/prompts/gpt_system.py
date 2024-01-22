import os

import yaml

from memgpt.constants import MEMGPT_DIR
from memgpt.prompts.prompt_template import PromptTemplate


def get_system_text(key):
    filename = f"{key}.txt"
    file_path = os.path.join(os.path.dirname(__file__), "system", filename)
    system_message = ""
    # first look in prompts/system/*.txt
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            system_message = file.read().strip()
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
                system_message = file.read().strip()
        else:
            raise FileNotFoundError(f"No file found for key {key}, path={file_path}")

    if not key.endswith("_templated"):
        return {"system_message": system_message, "template": "", "template_fields": {}}
    else:
        default_fields_yaml_filename = f"default_template_fields.yaml"
        default_fields_yaml_file_path = os.path.join(os.path.dirname(__file__), "system", default_fields_yaml_filename)
        if os.path.exists(default_fields_yaml_file_path):
            with open(default_fields_yaml_file_path, "r") as file:
                default_template_fields = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"No default template fields file found for key {key}, path={default_fields_yaml_file_path}")

        yaml_filename = filename.replace("_templated.txt", "_templated.yaml")
        template_fields_yaml_file_path = file_path.replace(filename, yaml_filename)
        template_fields = {}
        if os.path.exists(template_fields_yaml_file_path):
            with open(template_fields_yaml_file_path, "r") as file:
                template_fields = yaml.safe_load(file)

        for field, content in default_template_fields.items():
            if field not in template_fields:
                template_fields[field] = content

        template = PromptTemplate.from_string(system_message)
        return {"system_message": template.generate_prompt(template_fields), "template": system_message, "template_fields": template_fields}
