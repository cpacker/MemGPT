from .prompts import gpt_functions
from .prompts import gpt_system
from .agent import AgentAsync
from .utils import printd


DEFAULT = "memgpt_chat"


def use_preset(preset_name, model, persona, human, interface, persistence_manager):
    """Storing combinations of SYSTEM + FUNCTION prompts"""

    if preset_name == "memgpt_chat":
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
        available_functions = [v for k, v in gpt_functions.FUNCTIONS_CHAINING.items() if k in functions]
        printd(f"Available functions:\n", [x["name"] for x in available_functions])
        assert len(functions) == len(available_functions)

        if "gpt-3.5" in model:
            # use a different system message for gpt-3.5
            preset_name = "memgpt_gpt35_extralong"

        return AgentAsync(
            model=model,
            system=gpt_system.get_system_text(preset_name),
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
