from typing import List
import importlib
import inspect
import os
import uuid

from memgpt.data_types import AgentState, Preset
from memgpt.functions.functions import load_function_set
from memgpt.interface import AgentInterface
from memgpt.metadata import MetadataStore
from memgpt.schemas.block import Human, Persona
from memgpt.schemas.tool import Tool
from memgpt.presets.utils import load_all_presets
from memgpt.utils import list_human_files, list_persona_files, printd

available_presets = load_all_presets()
preset_options = list(available_presets.keys())


def load_module_tools(module_name="base"):
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
            Tool(
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
    for tool in load_module_tools(module_name="base"):
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
        persona = Persona(name=name, text=text, user_id=user_id)
        ms.add_persona(persona)
    for human_file in list_human_files():
        text = open(human_file, "r", encoding="utf-8").read()
        name = os.path.basename(human_file).replace(".txt", "")
        if ms.get_human(user_id=user_id, name=name) is not None:
            printd(f"Human '{name}' already exists for user '{user_id}'")
            continue
        human = Human(name=name, text=text, user_id=user_id)
        print(human, user_id)
        ms.add_human(human)


# def create_agent_from_preset(preset_name, agent_config, model, persona, human, interface, persistence_manager):
def create_agent_from_preset(
    agent_state: AgentState, preset: Preset, interface: AgentInterface, persona_is_file: bool = True, human_is_file: bool = True
):
    """Initialize a new agent from a preset (combination of system + function)"""
    raise DeprecationWarning("Function no longer supported - pass a Preset object to Agent.__init__ instead")

def create_preset_from_file(filename: str, name: str, user_id: uuid.UUID, ms: MetadataStore) -> Preset:
    preset_config = load_yaml_file(filename)
    preset_system_prompt = preset_config["system_prompt"]
    preset_function_set_names = preset_config["functions"]
    functions_schema = generate_functions_json(preset_function_set_names)

    if ms.get_preset(user_id=user_id, name=name) is not None:
        printd(f"Preset '{name}' already exists for user '{user_id}'")
        return ms.get_preset(user_id=user_id, name=name)

    preset = Preset(
        user_id=user_id,
        name=name,
        system=gpt_system.get_system_text(preset_system_prompt),
        persona=get_persona_text(settings.persona),
        human=get_human_text(settings.human),
        persona_name=settings.persona,
        human_name=settings.human,
        functions_schema=functions_schema,
    )
    ms.create_preset(preset)
    return preset


def load_preset(preset_name: str, user_id: uuid.UUID):
    preset_config = available_presets[preset_name]
    preset_system_prompt = preset_config["system_prompt"]
    preset_function_set_names = preset_config["functions"]
    functions_schema = generate_functions_json(preset_function_set_names)

    preset = Preset(
        user_id=user_id,
        name=preset_name,
        system=gpt_system.get_system_text(preset_system_prompt),
        persona=get_persona_text(settings.persona),
        persona_name=settings.persona,
        human=get_human_text(settings.human),
        human_name=settings.human,
        functions_schema=functions_schema,
    )
    return preset


def add_default_presets(user_id: uuid.UUID, ms: MetadataStore):
    """Add the default presets to the metadata store"""
    # make sure humans/personas added
    add_default_humans_and_personas(user_id=user_id, ms=ms)

    # make sure base functions added
    # TODO: pull from functions instead
    add_default_tools(user_id=user_id, ms=ms)

    # add default presets
    for preset_name in preset_options:
        if ms.get_preset(user_id=user_id, name=preset_name) is not None:
            printd(f"Preset '{preset_name}' already exists for user '{user_id}'")
            continue

        preset = load_preset(preset_name, user_id)
        ms.create_preset(preset)


def generate_functions_json(preset_functions: List[str]):
    """
    Generate JSON schema for the functions based on what is locally available.

    TODO: store function definitions in the DB, instead of locally
    """
    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    available_functions = load_all_function_sets()
    # Filter down the function set based on what the preset requested
    preset_function_set = {}
    for f_name in preset_functions:
        if f_name not in available_functions:
            raise ValueError(f"Function '{f_name}' was specified in preset, but is not in function library:\n{available_functions.keys()}")
        preset_function_set[f_name] = available_functions[f_name]
    assert len(preset_functions) == len(preset_function_set)
    preset_function_set_schemas = [f_dict["json_schema"] for f_name, f_dict in preset_function_set.items()]
    printd(f"Available functions:\n", list(preset_function_set.keys()))
    return preset_function_set_schemas


# def create_agent_from_preset(preset_name, agent_config, model, persona, human, interface, persistence_manager):
def create_agent_from_preset(
    agent_state: AgentState, preset: Preset, interface: AgentInterface, persona_is_file: bool = True, human_is_file: bool = True
):
    """Initialize a new agent from a preset (combination of system + function)"""
    raise DeprecationWarning("Function no longer supported - pass a Preset object to Agent.__init__ instead")
