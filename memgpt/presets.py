from .prompts import gpt_functions
from .prompts import gpt_system
from .functions.functions import load_all_function_sets

DEFAULT_PRESET = "memgpt_chat"
preset_options = [DEFAULT_PRESET]


def use_preset(preset_name, agent_config, model, persona, human, interface, persistence_manager):
    """Storing combinations of SYSTEM + FUNCTION prompts"""

    from memgpt.agent import Agent
    from memgpt.utils import printd

    # For each function
    available_function_sets = load_all_function_sets()
    # available_functions = {}
    # for function_set in available_functions_sets.values():
    #     available_functions.update(function_set)
    merged_functions = {}
    for set_name, function_set in available_function_sets.items():
        for function_name, function_info in function_set.items():
            if function_name in merged_functions:
                raise ValueError(f"Duplicate function name '{function_name}' found in function set '{set_name}'")
            merged_functions[function_name] = function_info

    # Available functions is a mapping from:
    # function_name -> {
    #   json_schema: schema
    #   python_function: function
    # }
    available_functions = merged_functions

    if preset_name == DEFAULT_PRESET:
        functions = [
            "send_message",
            "pause_heartbeats",
            "core_memory_append",
            "core_memory_replace",
            "conversation_search",
            "conversation_search_date",
            "archival_memory_insert",
            "archival_memory_search",
        ]
        # available_functions = [v for k, v in gpt_functions.FUNCTIONS_CHAINING.items() if k in functions]
        agent_function_set = {f_name: f_dict for f_name, f_dict in available_functions.items() if f_name in functions}
        printd(f"Available functions:\n", [f_name for f_name, f_dict in agent_function_set.items()])
        assert len(functions) == len(agent_function_set)

        if "gpt-3.5" in model:
            # use a different system message for gpt-3.5
            preset_name = "memgpt_gpt35_extralong"

        return Agent(
            config=agent_config,
            model=model,
            system=gpt_system.get_system_text(preset_name),
            functions=agent_function_set,
            interface=interface,
            persistence_manager=persistence_manager,
            persona_notes=persona,
            human_notes=human,
            # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
            first_message_verify_mono=True if "gpt-4" in model else False,
        )

    elif preset_name == "memgpt_extras":
        functions = [
            "send_message",
            "pause_heartbeats",
            "core_memory_append",
            "core_memory_replace",
            "conversation_search",
            "conversation_search_date",
            "archival_memory_insert",
            "archival_memory_search",
            # extra for read/write to files
            "read_from_text_file",
            "append_to_text_file",
            # internet access
            "http_request",
        ]
        available_functions = [v for k, v in gpt_functions.FUNCTIONS_CHAINING.items() if k in functions]
        printd(f"Available functions:\n", [x["name"] for x in available_functions])
        assert len(functions) == len(available_functions)

        if "gpt-3.5" in model:
            # use a different system message for gpt-3.5
            preset_name = "memgpt_gpt35_extralong"

        return Agent(
            model=model,
            system=gpt_system.get_system_text("memgpt_chat"),
            functions=available_functions,
            interface=interface,
            persistence_manager=persistence_manager,
            persona_notes=persona,
            human_notes=human,
            # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
            first_message_verify_mono=True if "gpt-4" in model else False,
        )

    else:
        raise ValueError(preset_name)
