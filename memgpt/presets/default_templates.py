default_system_message_layout_template = """{system}

### Memory [last modified: {memory_edit_timestamp}]
{len_recall_memory} previous messages between you and the user are stored in recall memory (use functions to access them)
{len_archival_memory} total memories you created are stored in archival memory (use functions to access them)

Core memory shown below (limited in size, additional information stored in archival / recall memory):

{core_memory_content}"""

default_core_memory_section_template = """<{memory_key} characters={memory_value_length}/{memory_value_limit}>

{memory_value}

</{memory_key}>
"""

default_system_message_template = """{memgpt_introduction}

{memgpt_realism_authenticity}

{memgpt_control_flow}

{memgpt_basic_functions}

{memgpt_memory_editing}

{memgpt_recall_memory}

{memgpt_core_memory}

{memgpt_archival_memory}

{memgpt_introduction_end}"""
