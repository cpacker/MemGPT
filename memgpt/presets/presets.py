import importlib
import os
import uuid
from typing import List

from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA
from memgpt.data_types import AgentState, Preset
from memgpt.functions.functions import load_all_function_sets, load_function_set
from memgpt.interface import AgentInterface
from memgpt.metadata import MetadataStore
from memgpt.models.pydantic_models import HumanModel, PersonaModel, ToolModel
from memgpt.presets.utils import load_all_presets, load_yaml_file
from memgpt.prompts import gpt_system
from memgpt.utils import (
    get_human_text,
    get_persona_text,
    list_human_files,
    list_persona_files,
    printd,
)

available_presets = load_all_presets()
preset_options = list(available_presets.keys())


def add_default_tools(user_id: uuid.UUID, ms: MetadataStore):
    module_name = "base"
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

    from pprint import pprint

    print("BASE FUNCTIONS", functions_to_schema.keys())
    pprint(functions_to_schema)

    # create tool in db
    for name, schema in functions_to_schema.items():
        ms.add_tool(ToolModel(name=name, tags=["base"], source_type="python", json_schema=schema["json_schema"]))


def add_default_humans_and_personas(user_id: uuid.UUID, ms: MetadataStore):
    for persona_file in list_persona_files():
        text = open(persona_file, "r").read()
        name = os.path.basename(persona_file).replace(".txt", "")
        if ms.get_persona(user_id=user_id, name=name) is not None:
            printd(f"Persona '{name}' already exists for user '{user_id}'")
            continue
        persona = PersonaModel(name=name, text=text, user_id=user_id)
        ms.add_persona(persona)
    for human_file in list_human_files():
        text = open(human_file, "r").read()
        name = os.path.basename(human_file).replace(".txt", "")
        if ms.get_human(user_id=user_id, name=name) is not None:
            printd(f"Human '{name}' already exists for user '{user_id}'")
            continue
        human = HumanModel(name=name, text=text, user_id=user_id)
        ms.add_human(human)


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
        persona=get_persona_text(DEFAULT_PERSONA),
        human=get_human_text(DEFAULT_HUMAN),
        persona_name=DEFAULT_PERSONA,
        human_name=DEFAULT_HUMAN,
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
        persona=get_persona_text(DEFAULT_PERSONA),
        persona_name=DEFAULT_PERSONA,
        human=get_human_text(DEFAULT_HUMAN),
        human_name=DEFAULT_HUMAN,
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

    # Input validation
    if agent_state.persona is None:
        raise ValueError(f"'persona' not specified in AgentState (required)")
    if agent_state.human is None:
        raise ValueError(f"'human' not specified in AgentState (required)")
    if agent_state.preset is None:
        raise ValueError(f"'preset' not specified in AgentState (required)")
    if not (agent_state.state == {} or agent_state.state is None):
        raise ValueError(f"'state' must be uninitialized (empty)")

    assert preset is not None, "preset cannot be none"
    preset_name = agent_state.preset
    assert preset_name == preset.name, f"AgentState preset '{preset_name}' does not match preset name '{preset.name}'"
    persona = agent_state.persona
    human = agent_state.human
    model = agent_state.llm_config.model

    from memgpt.agent import Agent

    # available_presets = load_all_presets()
    # if preset_name not in available_presets:
    #    raise ValueError(f"Preset '{preset_name}.yaml' not found")
    # preset = available_presets[preset_name]
    # preset_system_prompt = preset["system_prompt"]
    # preset_function_set_names = preset["functions"]
    # preset_function_set_schemas = generate_functions_json(preset_function_set_names)
    # Override the following in the AgentState:
    #   persona: str  # the current persona text
    #   human: str  # the current human text
    #   system: str,  # system prompt (not required if initializing with a preset)
    #   functions: dict,  # schema definitions ONLY (function code linked at runtime)
    #   messages: List[dict],  # in-context messages
    agent_state.state = {
        "persona": get_persona_text(persona) if persona_is_file else persona,
        "human": get_human_text(human) if human_is_file else human,
        "system": preset.system,
        "functions": preset.functions_schema,
        "messages": None,
    }

    return Agent(
        agent_state=agent_state,
        interface=interface,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=True if (model is not None and "gpt-4" in model) else False,
    )
