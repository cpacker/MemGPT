import importlib
import inspect
import os
import uuid

from memgpt.data_types import AgentState, Preset
from memgpt.functions.functions import load_function_set
from memgpt.interface import AgentInterface
from memgpt.metadata import MetadataStore
from memgpt.models.pydantic_models import HumanModel, PersonaModel, ToolModel
from memgpt.presets.utils import load_all_presets
from memgpt.utils import list_human_files, list_persona_files, printd

available_presets = load_all_presets()
preset_options = list(available_presets.keys())


def load_module_tools(module_name="base"):
    # return List[ToolModel] from base.py tools
    full_module_name = f"memgpt.functions.function_sets.{module_name}"
    try:
        module = importlib.import_module(full_module_name)
    except Exception as e:
        # Handle other general exceptions
        raise e

    # function tags

    try:
        # Load the function set
        functions_to_schema = load_function_set(module)
    except ValueError as e:
        err = f"Error loading function set '{module_name}': {e}"
        printd(err)

    # create tool in db
    tools = []
    for name, schema in functions_to_schema.items():
        # print([str(inspect.getsource(line)) for line in schema["imports"]])
        source_code = inspect.getsource(schema["python_function"])
        tags = [module_name]
        if module_name == "base":
            tags.append("memgpt-base")

        tools.append(
            ToolModel(
                name=name,
                tags=tags,
                source_type="python",
                module=schema["module"],
                source_code=source_code,
                json_schema=schema["json_schema"],
            )
        )
    return tools


def add_default_tools(user_id: uuid.UUID, ms: MetadataStore):
    module_name = "base"
    for tool in load_module_tools(module_name=module_name):
        existing_tool = ms.get_tool(tool.name)
        if not existing_tool:
            ms.add_tool(tool)


def add_default_humans_and_personas(user_id: uuid.UUID, ms: MetadataStore):
    for persona_file in list_persona_files():
        text = open(persona_file, "r", encoding="utf-8").read()
        name = os.path.basename(persona_file).replace(".txt", "")
        if ms.get_persona(user_id=user_id, name=name) is not None:
            printd(f"Persona '{name}' already exists for user '{user_id}'")
            continue
        persona = PersonaModel(name=name, text=text, user_id=user_id)
        ms.add_persona(persona)
    for human_file in list_human_files():
        text = open(human_file, "r", encoding="utf-8").read()
        name = os.path.basename(human_file).replace(".txt", "")
        if ms.get_human(user_id=user_id, name=name) is not None:
            printd(f"Human '{name}' already exists for user '{user_id}'")
            continue
        human = HumanModel(name=name, text=text, user_id=user_id)
        print(human, user_id)
        ms.add_human(human)


# def create_agent_from_preset(preset_name, agent_config, model, persona, human, interface, persistence_manager):
def create_agent_from_preset(
    agent_state: AgentState, preset: Preset, interface: AgentInterface, persona_is_file: bool = True, human_is_file: bool = True
):
    """Initialize a new agent from a preset (combination of system + function)"""
    raise DeprecationWarning("Function no longer supported - pass a Preset object to Agent.__init__ instead")
