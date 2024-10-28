import ast
import os
from typing import Any, Optional

from e2b_code_interpreter import Sandbox

from letta.schemas.tool import Tool


class SecureExecutionEnvironment:
    DIR = "/home/user/"

    def __init__(self, tool: Tool, args: dict):
        self.assert_required_args(tool, args)
        self.tool = tool
        self.args = args

    def run(self) -> Optional[Any]:
        code = self.generate_execution_script()

        sbx = Sandbox()
        sbx.files.write(f"{SecureExecutionEnvironment.DIR}source.py", code)
        sbx.commands.run(
            f"pip install pipreqs && "
            f"pipreqs {SecureExecutionEnvironment.DIR} && "
            f"pip install -r {SecureExecutionEnvironment.DIR}requirements.txt"
        )

        execution = sbx.run_code(code, envs=self.get_envs())
        if execution.error is not None:
            raise Exception(execution.error)
        elif len(execution.results) == 0:
            function_response = None
        else:
            try:
                function_response = ast.literal_eval(execution.results[0].text)
            except SyntaxError:
                function_response = execution.results[0].text

        sbx.kill()
        return function_response

    def generate_execution_script(self) -> str:
        code = ""

        for param in self.args:
            code += self.initialize_param(param, self.args[param])

        code += "\n" + self.tool.source_code + "\n"

        code += self.invoke_function_call()

        return code

    def initialize_param(self, name: str, raw_value: str) -> str:
        params = self.tool.json_schema["parameters"]["properties"]
        spec = params.get(name)
        if spec is None:
            # ignore extra params (like 'self') for now
            return ""

        param_type = spec.get("type")
        if param_type is None and spec.get("parameters"):
            param_type = spec["parameters"].get("type")

        if param_type == "string":
            value = '"' + raw_value + '"'
        elif param_type == "integer" or param_type == "boolean":
            value = raw_value
        else:
            raise TypeError(f"unsupported type: {param_type}")

        return name + " = " + str(value) + "\n"

    def invoke_function_call(self) -> str:
        kwargs = []
        for name in self.args:
            if name in self.tool.json_schema["parameters"]["properties"]:
                kwargs.append(name)

        params = ", ".join([f"{arg}={arg}" for arg in kwargs])
        return self.tool.name + "(" + params + ")"

    def get_envs(self) -> dict:
        envs = {}

        # hardcode for now. need more info on ux to formalize this
        settings_dict = {
            "composio": ["COMPOSIO_API_KEY"],
            "langchain": [],
            "crew-ai": [],
        }

        for tag in self.tool.tags:
            settings = settings_dict.get(tag, [])
            for setting in settings:
                envs[setting] = os.environ.get(setting)

        return envs

    @staticmethod
    def assert_required_args(tool: Tool, args: dict):
        required_args = tool.json_schema["parameters"]["required"]
        if "request_heartbeat" in required_args:
            required_args.remove("request_heartbeat")

        available_args = list(args.keys())

        missing_args = [arg for arg in required_args if arg not in available_args]
        if missing_args:
            raise TypeError(f"{tool.name}() missing {len(missing_args)} required positional argument: '{', '.join(missing_args)}'")
