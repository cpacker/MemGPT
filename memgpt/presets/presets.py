from memgpt.data_types import AgentState
from memgpt.interface import AgentInterface
from memgpt.presets.utils import load_all_presets, is_valid_yaml_format
from memgpt.utils import get_human_text, get_persona_text
from memgpt.prompts import gpt_system
from memgpt.functions.functions import load_all_function_sets


available_presets = load_all_presets()
preset_options = list(available_presets.keys())


# def create_agent_from_preset(preset_name, agent_config, model, persona, human, interface, persistence_manager):
def create_agent_from_preset(agent_state: AgentState, interface: AgentInterface):
    """Initialize a new agent from a preset (combination of system + function)"""

    # Input validation
    if agent_state.persona is None:
        raise ValueError(f"'persona' not specified in AgentState (required)")
    if agent_state.human is None:
        raise ValueError(f"'human' not specified in AgentState (required)")
    if agent_state.preset is None:
        raise ValueError(f"'preset' not specified in AgentState (required)")
    if not (agent_state.state == {} or agent_state.state is None):
        raise ValueError(f"'state' must be uninitialized (empty)")

    preset_name = agent_state.preset
    persona_file = agent_state.persona
    human_file = agent_state.human
    model = agent_state.llm_config.model

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
    preset_function_set_schemas = [f_dict["json_schema"] for f_name, f_dict in preset_function_set.items()]
    printd(f"Available functions:\n", list(preset_function_set.keys()))

    # Override the following in the AgentState:
    #   persona: str  # the current persona text
    #   human: str  # the current human text
    #   system: str,  # system prompt (not required if initializing with a preset)
    #   functions: dict,  # schema definitions ONLY (function code linked at runtime)
    #   messages: List[dict],  # in-context messages
    agent_state.state = {
        "persona": get_persona_text(persona_file),
        "human": get_human_text(human_file),
        "system": gpt_system.get_system_text(preset_system_prompt),
        "functions": preset_function_set_schemas,
        "messages": None,
    }

    return Agent(
        agent_state=agent_state,
        interface=interface,
        # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
        first_message_verify_mono=True if (model is not None and "gpt-4" in model) else False,
    )
