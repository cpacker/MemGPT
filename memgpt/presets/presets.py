from typing import List
import os
from memgpt.data_types import AgentState, Preset
from memgpt.interface import AgentInterface
from memgpt.presets.utils import load_all_presets, is_valid_yaml_format, load_yaml_file
from memgpt.utils import get_human_text, get_persona_text, printd, list_human_files, list_persona_files
from memgpt.prompts import gpt_system
from memgpt.functions.functions import load_all_function_sets
from memgpt.metadata import MetadataStore
from memgpt.constants import DEFAULT_HUMAN, DEFAULT_PERSONA, DEFAULT_PRESET
from memgpt.models.pydantic_models import HumanModel, PersonaModel

import uuid


available_presets = load_all_presets()
preset_options = list(available_presets.keys())


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


def add_default_presets(user_id: uuid.UUID, ms: MetadataStore):
    """Add the default presets to the metadata store"""
    # make sure humans/personas added
    add_default_humans_and_personas(user_id=user_id, ms=ms)

    # add default presets
    for preset_name in preset_options:
        preset_config = available_presets[preset_name]
        preset_system_prompt = preset_config["system_prompt"]
        preset_function_set_names = preset_config["functions"]
        functions_schema = generate_functions_json(preset_function_set_names)

        if ms.get_preset(user_id=user_id, name=preset_name) is not None:
            printd(f"Preset '{preset_name}' already exists for user '{user_id}'")
            continue

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
