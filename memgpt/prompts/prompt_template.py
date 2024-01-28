import re
from dataclasses import dataclass
from typing import List, Dict, Union


class PromptTemplate:
    """
    Class representing a prompt template.

    Methods:
        generate_prompt(template_fields: dict, remove_empty_template_field=True) -> str:
        Generate a prompt by replacing placeholders in the template with values.

    Class Methods:
        from_string(template_string: str) -> PromptTemplate:
        Create a PromptTemplate from a string.
        from_file(template_file: str) -> PromptTemplate:
        Create a PromptTemplate from a file.

    Attributes:
        template (str): The template string containing placeholders.
    """

    def __init__(self, template_file=None, template_string=None):
        """
        Initialize a PromptTemplate instance.

        Args:
            template_file (str): The path to a file containing the template.
            template_string (str): The template string.
        """
        if template_file:
            with open(template_file, "r") as file:
                self.template = file.read()
        elif template_string:
            self.template = template_string
        else:
            raise ValueError("Either 'template_file' or 'template_string' must be provided")

    @classmethod
    def from_string(cls, template_string):
        """
        Create a PromptTemplate instance from a string.

        Args:
            template_string (str): The template string.

        Returns:
            PromptTemplate: Created PromptTemplate instance.
        """
        return cls(template_string=template_string)

    @classmethod
    def from_file(cls, template_file):
        """
        Create a PromptTemplate instance from a file.

        Args:
            template_file (str): The path to a file containing the template.

        Returns:
            PromptTemplate: Created PromptTemplate instance.
        """
        with open(template_file, "r") as file:
            template_string = file.read()
        return cls(template_string=template_string)

    @staticmethod
    def _remove_empty_placeholders(text):
        """
        Remove lines that contain only the empty placeholder.

        Args:
            text (str): The text containing placeholders.

        Returns:
            str: Text with empty placeholders removed.
        """
        # Remove lines that contain only the empty placeholder
        text = re.sub(rf'^{"__EMPTY_TEMPLATE_FIELD__"}$', "", text, flags=re.MULTILINE)
        # Remove the empty placeholder from lines with other content
        text = re.sub(rf'{"__EMPTY_TEMPLATE_FIELD__"}', "", text)
        return text

    def generate_prompt(self, template_fields: dict, remove_empty_template_field=True) -> str:
        """
        Generate a prompt by replacing placeholders in the template with values.

        Args:
            template_fields (dict): The template fields.
            remove_empty_template_field (bool): If True, removes lines with empty placeholders.

        Returns:
            str: The generated prompt.
        """
        cleaned_fields = {}
        for key, value in template_fields.items():
            cleaned_fields[key] = str(value) if not isinstance(value, str) else value

        template_fields = cleaned_fields
        if not remove_empty_template_field:

            def replace_placeholder(match):
                placeholder = match.group(1)
                return template_fields.get(placeholder, match.group(0))

            prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)
            return prompt

        def replace_placeholder(match):
            placeholder = match.group(1)
            if template_fields.get(placeholder, match.group(0)) != "":
                return template_fields.get(placeholder, match.group(0))
            return "__EMPTY_TEMPLATE_FIELD__"

        # Initial placeholder replacement
        prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)

        return self._remove_empty_placeholders(prompt)
