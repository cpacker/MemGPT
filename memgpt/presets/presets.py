from .utils import load_all_presets, is_valid_yaml_format
from ..prompts import gpt_functions
from ..prompts import gpt_system
from ..functions.functions import load_all_function_sets

DEFAULT_PRESET = "memgpt_chat"

available_presets = load_all_presets()
preset_options = list(available_presets.keys())


def use_preset(preset_name, agent_config, model, persona, human, interface, persistence_manager):
    """Storing combinations of SYSTEM + FUNCTION prompts"""

    from memgpt.agent import Agent
    from memgpt.utils import printd

    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    available_functions = load_all_function_sets()

    available_presets = load_all_presets()
    if preset_name not in available_presets:
        raise ValueError(f"Preset '{preset_name}.yaml' not found")

    preset = available_presets[preset_name]
    if not is_valid_yaml_format(preset, list(available_functions.keys())):
        raise ValueError(f"Preset '{preset_name}.yaml' is not valid")

    preset_system_prompt = preset["system_prompt"]
    preset_function_set_names = preset["functions"]

    # Filter down the function set based on what the preset requested
    preset_function_set = {}
    for f_name in preset_function_set_names:
        if f_name not in available_functions:
            raise ValueError(f"Function '{f_name}' was specified in preset, but is not in function library:\n{available_functions.keys()}")
        preset_function_set[f_name] = available_functions[f_name]
    assert len(preset_function_set_names) == len(preset_function_set)
    printd(f"Available functions:\n", list(preset_function_set.keys()))

    # preset_function_set = {f_name: f_dict for f_name, f_dict in available_functions.items() if f_name in preset_function_set_names}
    # printd(f"Available functions:\n", [f_name for f_name, f_dict in preset_function_set.items()])
    # Make sure that every function the preset wanted is inside the available functions
    # assert len(preset_function_set_names) == len(preset_function_set)

    return Agent(
        config=agent_config,
        model=model,
        system=gpt_system.get_system_text(preset_system_prompt),
        functions=preset_function_set,
        interface=interface,
        persistence_manager=persistence_manager,
        persona_notes=persona,
        human_notes=human,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=True if (model is not None and "gpt-4" in model) else False,
    )
