import ast
import base64
import os
import pickle
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Optional

from letta.log import get_logger
from letta.schemas.agent import AgentState
from letta.schemas.sandbox_config import SandboxConfig, SandboxRunResult, SandboxType
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.tool_manager import ToolManager
from letta.services.user_manager import UserManager
from letta.settings import tool_settings

logger = get_logger(__name__)


class ToolExecutionSandbox:
    METADATA_CONFIG_STATE_KEY = "config_state"
    REQUIREMENT_TXT_NAME = "requirements.txt"

    # For generating long, random marker hashes
    NAMESPACE = uuid.NAMESPACE_DNS
    LOCAL_SANDBOX_RESULT_START_MARKER = str(uuid.uuid5(NAMESPACE, "local-sandbox-result-start-marker"))
    LOCAL_SANDBOX_RESULT_END_MARKER = str(uuid.uuid5(NAMESPACE, "local-sandbox-result-end-marker"))

    def __init__(self, tool_name: str, args: dict, user_id: str, force_recreate=False):
        self.tool_name = tool_name
        self.args = args

        # Get the user
        # This user corresponds to the agent_state's user_id field
        # agent_state is the state of the agent that invoked this run
        self.user = UserManager().get_user_by_id(user_id=user_id)

        # Get the tool
        # TODO: So in theory, it's possible this retrieves a tool not provisioned to the agent
        # TODO: That would probably imply that agent_state is incorrectly configured
        self.tool = ToolManager().get_tool_by_name(tool_name=tool_name, actor=self.user)
        if not self.tool:
            raise ValueError(
                f"Agent attempted to invoke tool {self.tool_name} that does not exist for organization {self.user.organization_id}"
            )

        self.sandbox_config_manager = SandboxConfigManager(tool_settings)
        self.force_recreate = force_recreate

    def run(self, agent_state: Optional[AgentState] = None) -> Optional[SandboxRunResult]:
        """
        Run the tool in a sandbox environment.

        Args:
            agent_state (Optional[AgentState]): The state of the agent invoking the tool

        Returns:
            Tuple[Any, Optional[AgentState]]: Tuple containing (tool_result, agent_state)
        """
        if tool_settings.e2b_api_key:
            logger.info(f"Using e2b sandbox to execute {self.tool_name}")
            code = self.generate_execution_script(wrap_print_with_markers=False, agent_state=agent_state)
            result = self.run_e2b_sandbox(code=code)
        else:
            logger.info(f"Using local sandbox to execute {self.tool_name}")
            code = self.generate_execution_script(wrap_print_with_markers=True, agent_state=agent_state)
            result = self.run_local_dir_sandbox(code=code)

        # Log out any stdout from the tool run
        logger.info(f"Executed tool '{self.tool_name}', logging stdout from tool run: \n")
        for log_line in result.stdout:
            logger.info(f"{log_line}\n")
        logger.info(f"Ending stdout log from tool run.")

        # Return result
        return result

    # local sandbox specific functions

    def run_local_dir_sandbox(self, code: str) -> Optional[SandboxRunResult]:
        sbx_config = self.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.LOCAL, actor=self.user)
        local_configs = sbx_config.get_local_config()

        # Get environment variables for the sandbox
        # TODO: We set limit to 100 here, but maybe we want it uncapped? Realistically this should be fine.
        env_vars = self.sandbox_config_manager.get_sandbox_env_vars_as_dict(sandbox_config_id=sbx_config.id, actor=self.user, limit=100)

        # Prepare the environment for subprocess
        env = os.environ.copy()
        env.update(env_vars)

        # Safety checks
        # Check that sandbox_dir exists
        if not os.path.isdir(local_configs.sandbox_dir):
            raise FileNotFoundError(f"Sandbox directory does not exist: {local_configs.sandbox_dir}")

        # Write the code to a temp file in the sandbox_dir
        with tempfile.NamedTemporaryFile(mode="w", dir=local_configs.sandbox_dir, suffix=".py", delete=True) as temp_file:
            temp_file.write(code)
            temp_file.flush()  # Ensure all data is written to disk

            # Execute the code in a restricted subprocess
            try:
                print("ASDFASDFASDFR")
                print(sys.executable)
                result = subprocess.run(
                    [sys.executable, temp_file.name],  # Use the current Python interpreter
                    env=env,
                    cwd=local_configs.sandbox_dir,  # Restrict execution to sandbox_dir
                    timeout=60,
                    capture_output=True,
                    text=True,
                )
                if result.stderr:
                    logger.error(f"Sandbox execution error: {result.stderr}")
                    raise RuntimeError(f"Sandbox execution error: {result.stderr}")
                func_result, stdout = self.parse_out_function_results_markers(result.stdout)
                func_return, agent_state = self.parse_best_effort(func_result)
                return SandboxRunResult(
                    func_return=func_return,
                    agent_state=agent_state,
                    stdout=[stdout],
                    sandbox_config_fingerprint=sbx_config.fingerprint(),
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Executing tool {self.tool_name} has timed out.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Executing tool {self.tool_name} has process error: {e}")
            except Exception as e:
                raise RuntimeError(f"Executing tool {self.tool_name} has an unexpected error: {e}")

    def parse_out_function_results_markers(self, text: str):
        marker_len = len(self.LOCAL_SANDBOX_RESULT_START_MARKER)
        start_index = text.index(self.LOCAL_SANDBOX_RESULT_START_MARKER) + marker_len
        end_index = text.index(self.LOCAL_SANDBOX_RESULT_END_MARKER)
        return text[start_index:end_index], text[: start_index - marker_len] + text[end_index + +marker_len :]

    # e2b sandbox specific functions

    def run_e2b_sandbox(self, code: str) -> Optional[SandboxRunResult]:
        sbx_config = self.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.E2B, actor=self.user)
        sbx = self.get_running_e2b_sandbox_with_same_state(sbx_config)
        if not sbx or self.force_recreate:
            sbx = self.create_e2b_sandbox_with_metadata_hash(sandbox_config=sbx_config)

        # Since this sandbox was used, we extend its lifecycle by the timeout
        sbx.set_timeout(sbx_config.get_e2b_config().timeout)

        # Get environment variables for the sandbox
        # TODO: We set limit to 100 here, but maybe we want it uncapped? Realistically this should be fine.
        env_vars = self.sandbox_config_manager.get_sandbox_env_vars_as_dict(sandbox_config_id=sbx_config.id, actor=self.user, limit=100)
        execution = sbx.run_code(code, envs=env_vars)
        if execution.error is not None:
            raise Exception(f"Executing tool {self.tool_name} failed with {execution.error}. Generated code: \n\n{code}")
        elif len(execution.results) == 0:
            return None
        else:
            func_return, agent_state = self.parse_best_effort(execution.results[0].text)
            return SandboxRunResult(
                func_return=func_return,
                agent_state=agent_state,
                stdout=execution.logs.stdout,
                sandbox_config_fingerprint=sbx_config.fingerprint(),
            )

    def get_running_e2b_sandbox_with_same_state(self, sandbox_config: SandboxConfig) -> Optional["Sandbox"]:
        from e2b_code_interpreter import Sandbox

        # List running sandboxes and access metadata.
        running_sandboxes = self.list_running_e2b_sandboxes()

        # Hash the config to check the state
        state_hash = sandbox_config.fingerprint()
        for sandbox in running_sandboxes:
            if self.METADATA_CONFIG_STATE_KEY in sandbox.metadata and sandbox.metadata[self.METADATA_CONFIG_STATE_KEY] == state_hash:
                return Sandbox.connect(sandbox.sandbox_id)

        return None

    def create_e2b_sandbox_with_metadata_hash(self, sandbox_config: SandboxConfig) -> "Sandbox":
        from e2b_code_interpreter import Sandbox

        state_hash = sandbox_config.fingerprint()
        e2b_config = sandbox_config.get_e2b_config()
        if e2b_config.template:
            sbx = Sandbox(sandbox_config.get_e2b_config().template, metadata={self.METADATA_CONFIG_STATE_KEY: state_hash})
        else:
            # no template
            sbx = Sandbox(metadata={self.METADATA_CONFIG_STATE_KEY: state_hash}, **e2b_config.model_dump(exclude={"pip_requirements"}))

        # install pip requirements
        if e2b_config.pip_requirements:
            for package in e2b_config.pip_requirements:
                sbx.commands.run(f"pip install {package}")
        return sbx

    def list_running_e2b_sandboxes(self):
        from e2b_code_interpreter import Sandbox

        # List running sandboxes and access metadata.
        return Sandbox.list()

    # general utility functions

    def parse_best_effort(self, text: str) -> Any:
        result = pickle.loads(base64.b64decode(text))
        agent_state = None
        if not result["agent_state"] is None:
            agent_state = result["agent_state"]
        return result["results"], agent_state

    def parse_function_arguments(self, source_code: str, tool_name: str):
        """Get arguments of a function from its source code"""
        tree = ast.parse(source_code)
        args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == tool_name:
                for arg in node.args.args:
                    args.append(arg.arg)
        return args

    def generate_execution_script(self, agent_state: AgentState, wrap_print_with_markers: bool = False) -> str:
        """
        Generate code to run inside of execution sandbox.
        Passes into a serialized agent state into the code, to be accessed by the tool.

        Args:
            agent_state (AgentState): The agent state
            wrap_print_with_markers (bool): Whether to wrap print statements (?)

        Returns:
            code (str): The generated code strong
        """
        # dump JSON representation of agent state to re-load
        code = "from typing import *\n"
        code += "import pickle\n"
        code += "import sys\n"
        code += "import base64\n"

        # Load the agent state data into the program
        if agent_state:
            code += "import letta\n"
            code += "from letta import * \n"
            import pickle

            agent_state_pickle = pickle.dumps(agent_state)
            code += f"agent_state = pickle.loads({agent_state_pickle})\n"
        else:
            # agent state is None
            code += "agent_state = None\n"

        for param in self.args:
            code += self.initialize_param(param, self.args[param])

        if "agent_state" in self.parse_function_arguments(self.tool.source_code, self.tool.name):
            inject_agent_state = True
        else:
            inject_agent_state = False

        code += "\n" + self.tool.source_code + "\n"

        # TODO: handle wrapped print

        code += (
            'result = {"results": ' + self.invoke_function_call(inject_agent_state=inject_agent_state) + ', "agent_state": agent_state}\n'
        )
        code += "result = base64.b64encode(pickle.dumps(result)).decode('utf-8')\n"
        if wrap_print_with_markers:
            code += f"sys.stdout.write('{self.LOCAL_SANDBOX_RESULT_START_MARKER}')\n"
            code += f"sys.stdout.write(str(result))\n"
            code += f"sys.stdout.write('{self.LOCAL_SANDBOX_RESULT_END_MARKER}')\n"
        else:
            code += "result\n"

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

    def invoke_function_call(self, inject_agent_state: bool) -> str:
        """
        Generate the code string to call the function.

        Args:
            inject_agent_state (bool): Whether to inject the agent's state as an input into the tool

        Returns:
            str: Generated code string for calling the tool
        """
        kwargs = []
        for name in self.args:
            if name in self.tool.json_schema["parameters"]["properties"]:
                kwargs.append(name)

        param_list = [f"{arg}={arg}" for arg in kwargs]
        if inject_agent_state:
            param_list.append("agent_state=agent_state")
        params = ", ".join(param_list)
        # if "agent_state" in kwargs:
        #    params += ", agent_state=agent_state"
        # TODO: fix to figure out when to insert agent state or not
        # params += "agent_state=agent_state"

        func_call_str = self.tool.name + "(" + params + ")"
        return func_call_str

    #
