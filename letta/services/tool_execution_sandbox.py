import ast
import os
from typing import Any, Optional

from e2b_code_interpreter import Sandbox

from letta.schemas.tool import Tool
from letta.services.tool_manager import ToolManager
from letta.services.user_manager import UserManager


class ToolExecutionSandbox:
    DIR = "/home/user/"

    def __init__(self, tool_name: str, args: dict, user_id: str):
        from letta.server.server import db_context

        self.session_maker = db_context
        self.tool_name = tool_name
        self.args = args

        # Get the user
        # This user corresponds to the agent_state's user_id field
        # agent_state is the state of the agent that invoked this run
        self.user = UserManager().get_user_by_id(user_id=user_id)

        # Get the tool
        # TODO: So in theory, it's possible this retrieves a tool not provisioned to the agent
        # That would probably imply that agent_state is incorrectly configured
        self.tool = ToolManager().get_tool_by_name(tool_name=tool_name, actor=self.user)

    def run(self) -> Optional[Any]:
        if not self.tool:
            return f"Agent attempted to invoke tool {self.tool_name} that does not exist for organization {self.user.organization_id}"

        code = self.generate_execution_script()

        sbx = Sandbox()
        sbx.files.write(f"{ToolExecutionSandbox.DIR}source.py", code)
        sbx.commands.run(
            f"pip install pipreqs && "
            f"pipreqs {ToolExecutionSandbox.DIR} && "
            f"pip install -r {ToolExecutionSandbox.DIR}requirements.txt"
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

    def get_or_create_sandbox(self) -> Sandbox:
        Sandbox.list()

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
