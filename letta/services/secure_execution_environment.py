from typing import Optional

from e2b_code_interpreter import Sandbox

from letta.schemas.tool import Tool


class SecureExecutionEnvironment:

    def __init__(self, tool: Tool, args: dict):
        self.assert_required_args(tool, args)
        self.tool = tool
        self.args = args

    def run(self) -> Optional[str]:
        code = self.generate_execution_script()

        sbx = Sandbox()
        execution = sbx.run_code(code)

        if execution.error is not None:
            raise execution.error
        elif len(execution.results) == 0:
            function_response = None
        else:
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

        param_type = spec["type"]
        if param_type == "string":
            value = '"' + raw_value + '"'
        elif param_type == "integer" or param_type == "boolean":
            value = raw_value
        else:
            raise TypeError(f"Unsupported type: {param_type}.")

        return name + " = " + value + "\n"

    def invoke_function_call(self) -> str:
        kwargs = []
        for name in self.args:
            if name in self.tool.json_schema["parameters"]["properties"]:
                kwargs.append(name)

        params = ", ".join([f"{arg}={arg}" for arg in kwargs])
        return self.tool.name + "(" + params + ")"

    @staticmethod
    def assert_required_args(tool: Tool, args: dict):
        required_args = tool.json_schema["parameters"]["required"]
        if "request_heartbeat" in required_args:
            required_args.remove("request_heartbeat")

        available_args = list(args.keys())

        missing_args = [arg for arg in required_args if arg not in available_args]
        if missing_args:
            raise TypeError(f"{tool.name}() missing {len(missing_args)} required positional argument: '{', '.join(missing_args)}'")
