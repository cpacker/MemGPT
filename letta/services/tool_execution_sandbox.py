import ast
import base64
import io
import os
import pickle
import runpy
import subprocess
import sys
import tempfile
import traceback
import uuid
import venv
from typing import Any, Dict, Optional

from letta.log import get_logger
from letta.schemas.agent import AgentState
from letta.schemas.sandbox_config import SandboxConfig, SandboxRunResult, SandboxType
from letta.schemas.tool import Tool
from letta.schemas.user import User
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.tool_manager import ToolManager
from letta.settings import tool_settings
from letta.utils import get_friendly_error_msg

logger = get_logger(__name__)


class ToolExecutionSandbox:
    METADATA_CONFIG_STATE_KEY = "config_state"
    REQUIREMENT_TXT_NAME = "requirements.txt"

    # For generating long, random marker hashes
    NAMESPACE = uuid.NAMESPACE_DNS
    LOCAL_SANDBOX_RESULT_START_MARKER = str(uuid.uuid5(NAMESPACE, "local-sandbox-result-start-marker"))
    LOCAL_SANDBOX_RESULT_END_MARKER = str(uuid.uuid5(NAMESPACE, "local-sandbox-result-end-marker"))

    # This is the variable name in the auto-generated code that contains the function results
    # We make this a long random string to avoid collisions with any variables in the user's code
    LOCAL_SANDBOX_RESULT_VAR_NAME = "result_ZQqiequkcFwRwwGQMqkt"

    def __init__(self, tool_name: str, args: dict, user: User, force_recreate=False, tool_object: Optional[Tool] = None):
        self.tool_name = tool_name
        self.args = args
        self.user = user

        # If a tool object is provided, we use it directly, otherwise pull via name
        if tool_object is not None:
            self.tool = tool_object
        else:
            # Get the tool via name
            # TODO: So in theory, it's possible this retrieves a tool not provisioned to the agent
            # TODO: That would probably imply that agent_state is incorrectly configured
            self.tool = ToolManager().get_tool_by_name(tool_name=tool_name, actor=self.user)
            if not self.tool:
                raise ValueError(
                    f"Agent attempted to invoke tool {self.tool_name} that does not exist for organization {self.user.organization_id}"
                )

        self.sandbox_config_manager = SandboxConfigManager(tool_settings)
        self.force_recreate = force_recreate

    def run(self, agent_state: Optional[AgentState] = None) -> SandboxRunResult:
        """
        Run the tool in a sandbox environment.

        Args:
            agent_state (Optional[AgentState]): The state of the agent invoking the tool

        Returns:
            Tuple[Any, Optional[AgentState]]: Tuple containing (tool_result, agent_state)
        """
        if tool_settings.e2b_api_key:
            logger.debug(f"Using e2b sandbox to execute {self.tool_name}")
            result = self.run_e2b_sandbox(agent_state=agent_state)
        else:
            logger.debug(f"Using local sandbox to execute {self.tool_name}")
            result = self.run_local_dir_sandbox(agent_state=agent_state)

        # Log out any stdout/stderr from the tool run
        logger.debug(f"Executed tool '{self.tool_name}', logging output from tool run: \n")
        for log_line in (result.stdout or []) + (result.stderr or []):
            logger.debug(f"{log_line}")
        logger.debug(f"Ending output log from tool run.")

        # Return result
        return result

    # local sandbox specific functions
    from contextlib import contextmanager

    @contextmanager
    def temporary_env_vars(self, env_vars: dict):
        original_env = os.environ.copy()  # Backup original environment variables
        os.environ.update(env_vars)  # Update with the new variables
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(original_env)  # Restore original environment variables

    def run_local_dir_sandbox(self, agent_state: AgentState) -> SandboxRunResult:
        sbx_config = self.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.LOCAL, actor=self.user)
        local_configs = sbx_config.get_local_config()

        # Get environment variables for the sandbox
        env_vars = self.sandbox_config_manager.get_sandbox_env_vars_as_dict(sandbox_config_id=sbx_config.id, actor=self.user, limit=100)
        env = os.environ.copy()
        env.update(env_vars)

        # Safety checks
        if not os.path.isdir(local_configs.sandbox_dir):
            raise FileNotFoundError(f"Sandbox directory does not exist: {local_configs.sandbox_dir}")

        # Write the code to a temp file in the sandbox_dir
        with tempfile.NamedTemporaryFile(mode="w", dir=local_configs.sandbox_dir, suffix=".py", delete=False) as temp_file:
            if local_configs.use_venv:
                # If using venv, we need to wrap with special string markers to separate out the output and the stdout (since it is all in stdout)
                code = self.generate_execution_script(agent_state=agent_state, wrap_print_with_markers=True)
            else:
                code = self.generate_execution_script(agent_state=agent_state)

            temp_file.write(code)
            temp_file.flush()
            temp_file_path = temp_file.name

        try:
            if local_configs.use_venv:
                return self.run_local_dir_sandbox_venv(sbx_config, env, temp_file_path)
            else:
                return self.run_local_dir_sandbox_runpy(sbx_config, env_vars, temp_file_path)
        except Exception as e:
            logger.error(f"Executing tool {self.tool_name} has an unexpected error: {e}")
            logger.error(f"Logging out tool {self.tool_name} auto-generated code for debugging: \n\n{code}")
            raise e
        finally:
            # Clean up the temp file
            os.remove(temp_file_path)

    def run_local_dir_sandbox_venv(self, sbx_config: SandboxConfig, env: Dict[str, str], temp_file_path: str) -> SandboxRunResult:
        local_configs = sbx_config.get_local_config()
        venv_path = os.path.join(local_configs.sandbox_dir, local_configs.venv_name)

        # Safety checks for the venv: verify that the venv path exists and is a directory
        if not os.path.isdir(venv_path):
            logger.warning(f"Virtual environment directory does not exist at: {venv_path}, creating one now...")
            self.create_venv_for_local_sandbox(sandbox_dir_path=local_configs.sandbox_dir, venv_path=venv_path, env=env)

        # Ensure the python interpreter exists in the virtual environment
        python_executable = os.path.join(venv_path, "bin", "python3")
        if not os.path.isfile(python_executable):
            raise FileNotFoundError(f"Python executable not found in virtual environment: {python_executable}")

        # Set up env for venv
        env["VIRTUAL_ENV"] = venv_path
        env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
        # Suppress all warnings
        env["PYTHONWARNINGS"] = "ignore"

        # Execute the code in a restricted subprocess
        try:
            result = subprocess.run(
                [os.path.join(venv_path, "bin", "python3"), temp_file_path],
                env=env,
                cwd=local_configs.sandbox_dir,  # Restrict execution to sandbox_dir
                timeout=60,
                capture_output=True,
                text=True,
            )
            func_result, stdout = self.parse_out_function_results_markers(result.stdout)
            func_return, agent_state = self.parse_best_effort(func_result)
            return SandboxRunResult(
                func_return=func_return,
                agent_state=agent_state,
                stdout=[stdout] if stdout else [],
                stderr=[result.stderr] if result.stderr else [],
                status="success",
                sandbox_config_fingerprint=sbx_config.fingerprint(),
            )

        except subprocess.CalledProcessError as e:
            logger.error(f"Executing tool {self.tool_name} has process error: {e}")
            func_return = get_friendly_error_msg(
                function_name=self.tool_name,
                exception_name=type(e).__name__,
                exception_message=str(e),
            )
            return SandboxRunResult(
                func_return=func_return,
                agent_state=None,
                stdout=[e.stdout] if e.stdout else [],
                stderr=[e.stderr] if e.stderr else [],
                status="error",
                sandbox_config_fingerprint=sbx_config.fingerprint(),
            )

        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Executing tool {self.tool_name} has timed out.")

        except Exception as e:
            logger.error(f"Executing tool {self.tool_name} has an unexpected error: {e}")
            raise e

    def run_local_dir_sandbox_runpy(self, sbx_config: SandboxConfig, env_vars: Dict[str, str], temp_file_path: str) -> SandboxRunResult:
        status = "success"
        agent_state, stderr = None, None

        # Redirect stdout and stderr to capture script output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_stdout, captured_stderr = io.StringIO(), io.StringIO()
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        try:
            # Execute the temp file
            with self.temporary_env_vars(env_vars):
                result = runpy.run_path(temp_file_path, init_globals=env_vars)

            # Fetch the result
            func_result = result.get(self.LOCAL_SANDBOX_RESULT_VAR_NAME)
            func_return, agent_state = self.parse_best_effort(func_result)

        except Exception as e:
            func_return = get_friendly_error_msg(function_name=self.tool_name, exception_name=type(e).__name__, exception_message=str(e))
            traceback.print_exc(file=sys.stderr)
            status = "error"

        # Restore stdout and stderr and collect captured output
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        stdout_output = [captured_stdout.getvalue()] if captured_stdout.getvalue() else []
        stderr_output = [captured_stderr.getvalue()] if captured_stderr.getvalue() else []

        return SandboxRunResult(
            func_return=func_return,
            agent_state=agent_state,
            stdout=stdout_output,
            stderr=stderr_output,
            status=status,
            sandbox_config_fingerprint=sbx_config.fingerprint(),
        )

    def parse_out_function_results_markers(self, text: str):
        if self.LOCAL_SANDBOX_RESULT_START_MARKER not in text:
            return "", text
        marker_len = len(self.LOCAL_SANDBOX_RESULT_START_MARKER)
        start_index = text.index(self.LOCAL_SANDBOX_RESULT_START_MARKER) + marker_len
        end_index = text.index(self.LOCAL_SANDBOX_RESULT_END_MARKER)
        return text[start_index:end_index], text[: start_index - marker_len] + text[end_index + +marker_len :]

    def create_venv_for_local_sandbox(self, sandbox_dir_path: str, venv_path: str, env: Dict[str, str]):
        # Step 1: Create the virtual environment
        venv.create(venv_path, with_pip=True)

        pip_path = os.path.join(venv_path, "bin", "pip")
        try:
            # Step 2: Upgrade pip
            logger.info("Upgrading pip in the virtual environment...")
            subprocess.run([pip_path, "install", "--upgrade", "pip"], env=env, check=True)

            # Step 3: Install packages from requirements.txt if provided
            requirements_txt_path = os.path.join(sandbox_dir_path, self.REQUIREMENT_TXT_NAME)
            if os.path.isfile(requirements_txt_path):
                logger.info(f"Installing packages from requirements file: {requirements_txt_path}")
                subprocess.run([pip_path, "install", "-r", requirements_txt_path], env=env, check=True)
                logger.info("Successfully installed packages from requirements.txt")
            else:
                logger.warning("No requirements.txt file provided or the file does not exist. Skipping package installation.")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error while setting up the virtual environment: {e}")
            raise RuntimeError(f"Failed to set up the virtual environment: {e}")

    # e2b sandbox specific functions

    def run_e2b_sandbox(self, agent_state: AgentState) -> SandboxRunResult:
        sbx_config = self.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.E2B, actor=self.user)
        sbx = self.get_running_e2b_sandbox_with_same_state(sbx_config)
        if not sbx or self.force_recreate:
            sbx = self.create_e2b_sandbox_with_metadata_hash(sandbox_config=sbx_config)

        # Since this sandbox was used, we extend its lifecycle by the timeout
        sbx.set_timeout(sbx_config.get_e2b_config().timeout)

        # Get environment variables for the sandbox
        # TODO: We set limit to 100 here, but maybe we want it uncapped? Realistically this should be fine.
        env_vars = self.sandbox_config_manager.get_sandbox_env_vars_as_dict(sandbox_config_id=sbx_config.id, actor=self.user, limit=100)
        code = self.generate_execution_script(agent_state=agent_state)
        execution = sbx.run_code(code, envs=env_vars)

        if execution.results:
            func_return, agent_state = self.parse_best_effort(execution.results[0].text)
        elif execution.error:
            logger.error(f"Executing tool {self.tool_name} failed with {execution.error}")
            func_return = get_friendly_error_msg(
                function_name=self.tool_name, exception_name=execution.error.name, exception_message=execution.error.value
            )
            execution.logs.stderr.append(execution.error.traceback)
        else:
            raise ValueError(f"Tool {self.tool_name} returned execution with None")

        return SandboxRunResult(
            func_return=func_return,
            agent_state=agent_state,
            stdout=execution.logs.stdout,
            stderr=execution.logs.stderr,
            status="error" if execution.error else "success",
            sandbox_config_fingerprint=sbx_config.fingerprint(),
        )

    def parse_exception_from_e2b_execution(self, e2b_execution: "Execution") -> Exception:
        builtins_dict = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
        # Dynamically fetch the exception class from builtins, defaulting to Exception if not found
        exception_class = builtins_dict.get(e2b_execution.error.name, Exception)
        return exception_class(e2b_execution.error.value)

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
        if not text:
            return None, None
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
            wrap_print_with_markers (bool): If true, we wrap the final statement with a `print` and wrap with special markers

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
            self.LOCAL_SANDBOX_RESULT_VAR_NAME
            + ' = {"results": '
            + self.invoke_function_call(inject_agent_state=inject_agent_state)
            + ', "agent_state": agent_state}\n'
        )
        code += (
            f"{self.LOCAL_SANDBOX_RESULT_VAR_NAME} = base64.b64encode(pickle.dumps({self.LOCAL_SANDBOX_RESULT_VAR_NAME})).decode('utf-8')\n"
        )

        if wrap_print_with_markers:
            code += f"sys.stdout.write('{self.LOCAL_SANDBOX_RESULT_START_MARKER}')\n"
            code += f"sys.stdout.write(str({self.LOCAL_SANDBOX_RESULT_VAR_NAME}))\n"
            code += f"sys.stdout.write('{self.LOCAL_SANDBOX_RESULT_END_MARKER}')\n"
        else:
            code += f"{self.LOCAL_SANDBOX_RESULT_VAR_NAME}\n"

        return code

    def _convert_param_to_value(self, param_type: str, raw_value: str) -> str:

        if param_type == "string":
            value = "pickle.loads(" + str(pickle.dumps(raw_value)) + ")"

        elif param_type == "integer" or param_type == "boolean" or param_type == "number":
            value = raw_value

        elif param_type == "array":
            value = raw_value

        elif param_type == "object":
            value = raw_value

        else:
            raise TypeError(f"Unsupported type: {param_type}, raw_value={raw_value}")
        return str(value)

    def initialize_param(self, name: str, raw_value: str) -> str:
        params = self.tool.json_schema["parameters"]["properties"]
        spec = params.get(name)
        if spec is None:
            # ignore extra params (like 'self') for now
            return ""

        param_type = spec.get("type")
        if param_type is None and spec.get("parameters"):
            param_type = spec["parameters"].get("type")

        value = self._convert_param_to_value(param_type, raw_value)

        return name + " = " + value + "\n"

    def invoke_function_call(self, inject_agent_state: bool) -> str:
        """
        Generate the code string to call the function.

        Args:
            inject_agent_state (bool): Whether to inject the axgent's state as an input into the tool

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
