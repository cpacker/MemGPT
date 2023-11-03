from .prompts import gpt_functions
from .prompts import gpt_system

from typing import List, Optional, Tuple
from dataclasses import dataclass, field

DEFAULT_PRESET = "memgpt_chat"
preset_options = [DEFAULT_PRESET]  # TODO: eventually remove
SYNC_CHAT = "memgpt_chat_sync"  # TODO: remove me after we move the CLI to AgentSync


@dataclass
class Preset:
    name: str = None
    pretty_name: str = None
    functions: Tuple[str] = ()
    sync: bool = False  # TODO: remove as a preset


@dataclass
class DefaultPreset(Preset):
    name: str = DEFAULT_PRESET
    pretty_name: str = "Default"
    functions: Tuple[str] = (
        "send_message",
        "pause_heartbeats",
        "core_memory_append",
        "core_memory_replace",
        "conversation_search",
        "conversation_search_date",
        "archival_memory_insert",
        "archival_memory_search",
    )


@dataclass
class SyncPreset(DefaultPreset):  # TODO: get rid of this
    name: str = SYNC_CHAT
    pretty_name: str = "Sync Chat"
    sync: bool = True


@dataclass
class ExtrasPreset(DefaultPreset):
    name: str = "memgpt_extras"
    pretty_name: str = "Extras (read/write to file)"

    def __init__(self):
        self.functions = super().functions + ("read_from_text_file", "append_to_text_file", "http_request")


presets = [DefaultPreset(), ExtrasPreset(), SyncPreset()]
preset_map = {preset.name: preset for preset in presets}


# def use_preset(preset_name, agent_config, model, persona, human, interface, persistence_manager):
def use_preset(agent_config, interface, persistence_manager):
    """Storing combinations of SYSTEM + FUNCTION prompts"""

    from memgpt.agent import AgentAsync, Agent
    from memgpt.utils import printd, get_human_text, get_persona_text

    # read config values
    preset_name = agent_config.preset
    model = agent_config.model
    persona = get_persona_text(agent_config.persona)
    human = get_human_text(agent_config.human)

    # setup functions
    assert preset_name in preset_map, f"Invalid preset name: {preset_name}"
    preset = preset_map[preset_name]
    functions = preset.functions
    available_functions = [v for k, v in gpt_functions.FUNCTIONS_CHAINING.items() if k in functions]
    printd(f"Available functions:\n", [x["name"] for x in available_functions])
    assert len(functions) == len(available_functions)

    # get system text name
    system_text = "memgpt_gpt35_extralong" if "gpt-3.5" in model else preset_name

    print(gpt_system.get_system_text(system_text), persona, human)

    if preset_name != "memgpt_chat_sync":
        return AgentAsync(
            config=agent_config,
            model=model,
            system=gpt_system.get_system_text(system_text),
            functions=available_functions,
            interface=interface,
            persistence_manager=persistence_manager,
            persona_notes=persona,
            human_notes=human,
            # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
            first_message_verify_mono=True if "gpt-4" in model else False,
        )
    else:
        return Agent(
            config=agent_config,
            model=model,
            system=gpt_system.get_system_text(DEFAULT_PRESET),
            functions=available_functions,
            interface=interface,
            persistence_manager=persistence_manager,
            persona_notes=persona,
            human_notes=human,
            # gpt-3.5-turbo tends to omit inner monologue, relax this requirement for now
            first_message_verify_mono=True if "gpt-4" in model else False,
        )
