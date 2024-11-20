import ast
import os
import subprocess
import tempfile
import venv
from typing import Any, Optional, Tuple

from letta.log import get_logger
from letta.schemas.agent import AgentState
from letta.schemas.sandbox_config import SandboxConfig, SandboxType
from letta.services.sandbox_config_manager import SandboxConfigManager
from letta.services.tool_manager import ToolManager
from letta.services.user_manager import UserManager
from letta.settings import tool_settings

logger = get_logger(__name__)


class ToolExecutionSandbox:
    DIR = "/home/user/"
    METADATA_CONFIG_STATE_KEY = "config_state"

    def __init__(self, tool_name: str, args: dict, user_id: str, force_recreate=False):
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
        # TODO: That would probably imply that agent_state is incorrectly configured
        self.tool = ToolManager().get_tool_by_name(tool_name=tool_name, actor=self.user)
        self.sandbox_config_manager = SandboxConfigManager()
        self.force_recreate = force_recreate

    def run(self, agent_state: Optional[AgentState] = None) -> Tuple[Any, Optional[AgentState]]:
        """
        Run the tool in a sandbox environment.

        Args:
            agent_state (Optional[AgentState]): The state of the agent invoking the tool

        Returns:
            Tuple[Any, Optional[AgentState]]: Tuple containing (tool_result, agent_state)
        """
        if not self.tool:
            return f"Agent attempted to invoke tool {self.tool_name} that does not exist for organization {self.user.organization_id}"
        if tool_settings.e2b_api_key:
            logger.info("Using e2b for tool execution...")
            code = self.generate_execution_script(wrap_print=False, agent_state=agent_state)
            return self.run_e2b_sandbox(code=code)
        else:
            logger.info("Using local sandbox for tool execution...")
            code = self.generate_execution_script(wrap_print=True, agent_state=agent_state)
            print(code)
            logger.info("Running code in local sandbox...", code)
            return self.run_local_dir_sandbox(code=code)

    def parse_results(self, results: str) -> Tuple[Any, Optional[AgentState]]:
        """Parse results string using ast"""
        result_data = self.ast_parse_best_effort(results)
        agent_state = None
        if result_data["agent_state"]:
            # agent_state = AgentState(**result_data["agent_state"])
            import pickle

            # load pickled agent state
            agent_state = pickle.loads(result_data["agent_state"])
        return result_data["results"], agent_state

    def run_local_dir_sandbox(self, code: str) -> Optional[Any]:
        sbx_config = self.sandbox_config_manager.get_or_create_default_sandbox_config(sandbox_type=SandboxType.LOCAL, actor=self.user)
        local_configs = sbx_config.get_local_config()

        # Get environment variables for the sandbox
        # TODO: We set limit to 100 here, but maybe we want it uncapped? Realistically this should be fine.
        env_vars = self.sandbox_config_manager.get_sandbox_env_vars_as_dict(sandbox_config_id=sbx_config.id, actor=self.user, limit=100)

        env = os.environ.copy()
        venv_path = os.path.join(local_configs.sandbox_dir, local_configs.venv_name)
        env["VIRTUAL_ENV"] = venv_path
        env["PATH"] = os.path.join(venv_path, "bin") + ":" + env["PATH"]
        env.update(env_vars)

        print("ENV", venv_path)

        # Safety checks
        # Check that sandbox_dir exists
        if not os.path.isdir(local_configs.sandbox_dir):
            raise FileNotFoundError(f"Sandbox directory does not exist: {local_configs.sandbox_dir}")
        # Verify that the venv path exists and is a directory
        if not os.path.isdir(venv_path):
            logger.warning(f"Virtual environment directory does not exist at: {venv_path}, creating one now...")
            venv.create(venv_path, with_pip=True)

        # Ensure the python interpreter exists in the virtual environment
        python_executable = os.path.join(venv_path, "bin", "python3")
        if not os.path.isfile(python_executable):
            raise FileNotFoundError(f"Python executable not found in virtual environment: {python_executable}")

        # Write the code to a temp file in the sandbox_dir
        with tempfile.NamedTemporaryFile(mode="w", dir=local_configs.sandbox_dir, suffix=".py", delete=True) as temp_file:
            temp_file.write(code)
            temp_file.flush()  # Ensure all data is written to disk

            # Execute the code in a restricted subprocess
            try:
                result = subprocess.run(
                    [os.path.join(venv_path, "bin", "python3"), temp_file.name],
                    env=env,
                    cwd=local_configs.sandbox_dir,  # Restrict execution to sandbox_dir
                    timeout=60,
                    capture_output=True,
                    text=True,
                )
                if result.stderr:
                    print(f"Sandbox execution error: {result.stderr}")
                    logger.error(f"Sandbox execution error: {result.stderr}")
                    raise RuntimeError(f"Sandbox execution error: {result.stderr}")
                print("RESULT LOCAL", result.stdout)
                # return self.ast_parse_best_effort(result.stdout)
                return self.parse_results(result.stdout)
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"Executing tool {self.tool_name} has timed out.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Executing tool {self.tool_name} has process error: {e}")
            except Exception as e:
                raise RuntimeError(f"Executing tool {self.tool_name} has an unexpected error: {e}")

    def run_e2b_sandbox(self, code: str) -> Optional[Any]:
        pass

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
        print(execution)
        print("exec results", execution.results)
        if execution.error is not None:
            raise Exception(f"Executing tool {self.tool_name} failed with {execution.error}")
        elif len(execution.results) == 0:
            function_response = None
        else:
            function_response = self.ast_parse_best_effort(execution.results[0].text)
        print("RESPONSE", function_response)

        # Note, we don't kill the sandbox
        return self.parse_results(function_response)

    def get_running_e2b_sandbox_with_same_state(self, sandbox_config: SandboxConfig) -> Optional["Sandbox"]:
        from e2b_code_interpreter import Sandbox

        # List running sandboxes and access metadata.
        running_sandboxes = self.list_running_e2b_sandboxes()

        # Hash the config to check the state
        state_hash = str(hash(sandbox_config))
        for sandbox in running_sandboxes:
            if self.METADATA_CONFIG_STATE_KEY in sandbox.metadata and sandbox.metadata[self.METADATA_CONFIG_STATE_KEY] == state_hash:
                return Sandbox.connect(sandbox.sandbox_id)

        return None

    def create_e2b_sandbox_with_metadata_hash(self, sandbox_config: SandboxConfig) -> "Sandbox":
        from e2b_code_interpreter import Sandbox

        state_hash = str(hash(sandbox_config))
        if sandbox_config.get_e2b_config().template_id:
            return Sandbox(sandbox_config.get_e2b_config().template_id, metadata={self.METADATA_CONFIG_STATE_KEY: state_hash})
        else:
            # no template
            return Sandbox(metadata={self.METADATA_CONFIG_STATE_KEY: state_hash}, **sandbox_config.config)

    def list_running_e2b_sandboxes(self):
        from e2b_code_interpreter import Sandbox

        # List running sandboxes and access metadata.
        return Sandbox.list()

    def ast_parse_best_effort(self, text: str) -> Any:
        try:
            result = ast.literal_eval(text)
        except SyntaxError:
            result = text
        except ValueError:
            result = text

        return result

    def parse_function_arguments(self, source_code: str, tool_name: str):
        """Get arguments of a function from its source code"""
        tree = ast.parse(source_code)
        args = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == tool_name:
                for arg in node.args.args:
                    args.append(arg.arg)
        return args

    def generate_execution_script(self, agent_state: AgentState, wrap_print: bool = False) -> str:
        """
        Generate code to run inside of execution sandbox.
        Passes into a serialized agent state into the code, to be accessed by the tool.

        Args:
            agent_state (AgentState): The agent state
            wrap_print (bool): Whether to wrap print statements (?)

        Returns:
            code (str): The generated code strong
        """
        # dump JSON representation of agent state to re-load
        code = "from typing import *\n"

        # Load the agent state data into the program
        if agent_state:
            # agent state is not None
            # agent_state_json = agent_state.model_dump_json()
            # code += f"agent_state_json = {agent_state_json}\n"
            # code += "agent_state = AgentState(**agent_state_json)\n"
            code += "import pickle\n"
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

        # function call string
        # code += self.invoke_function_call(agent_state=agent_state, wrap_print=wrap_print)
        # code += 'result_json = {"results": ' + self.invoke_function_call(agent_state=agent_state, wrap_print=wrap_print) + ', "agent_state": agent_state.model_dump_json() if agent_state else None}\n'
        code += (
            'result_json = {"results": '
            + self.invoke_function_call(inject_agent_state=inject_agent_state)
            + ', "agent_state": pickle.dumps(agent_state)}\n'
        )
        if wrap_print:
            code += "print(result_json)\n"
        else:
            code += "result_json\n"

        print(code)

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
