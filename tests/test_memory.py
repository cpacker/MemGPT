import pytest

# Import the classes here, assuming the above definitions are in a module named memory_module
from letta.schemas.memory import ChatMemory, Memory


@pytest.fixture
def sample_memory():
    return ChatMemory(persona="Chat Agent", human="User")


def test_create_chat_memory():
    """Test creating an instance of ChatMemory"""
    chat_memory = ChatMemory(persona="Chat Agent", human="User")
    assert chat_memory.get_block("persona").value == "Chat Agent"
    assert chat_memory.get_block("human").value == "User"


def test_dump_memory_as_json(sample_memory: Memory):
    """Test dumping ChatMemory as JSON compatible dictionary"""
    memory_dict = sample_memory.to_dict()["memory"]
    assert isinstance(memory_dict, dict)
    assert "persona" in memory_dict
    assert memory_dict["persona"]["value"] == "Chat Agent"


def test_load_memory_from_json(sample_memory: Memory):
    """Test loading ChatMemory from a JSON compatible dictionary"""
    memory_dict = sample_memory.to_dict()["memory"]
    print(memory_dict)
    new_memory = Memory.load(memory_dict)
    assert new_memory.get_block("persona").value == "Chat Agent"
    assert new_memory.get_block("human").value == "User"


def test_memory_limit_validation(sample_memory: Memory):
    """Test exceeding memory limit"""
    with pytest.raises(ValueError):
        ChatMemory(persona="x" * 3000, human="y" * 3000)

    with pytest.raises(ValueError):
        sample_memory.get_block("persona").value = "x" * 3000


def test_memory_jinja2_template_load(sample_memory: Memory):
    """Test loading a memory with and without a jinja2 template"""

    # Test loading a memory with a template
    memory_dict = sample_memory.to_dict()
    memory_dict["prompt_template"] = sample_memory.get_prompt_template()
    new_memory = Memory.load(memory_dict)
    assert new_memory.get_prompt_template() == sample_memory.get_prompt_template()

    # Test loading a memory without a template (old format)
    memory_dict = sample_memory.to_dict()
    memory_dict_old_format = memory_dict["memory"]
    new_memory = Memory.load(memory_dict_old_format)
    assert new_memory.get_prompt_template() is not None  # Ensure a default template is set
    assert new_memory.to_dict()["memory"] == memory_dict_old_format


def test_memory_jinja2_template(sample_memory: Memory):
    """Test to make sure the jinja2 template string is equivalent to the old __repr__ method"""

    def old_repr(self: Memory) -> str:
        """Generate a string representation of the memory in-context"""
        section_strs = []
        for section, module in self.memory.items():
            section_strs.append(f'<{section} characters="{len(module.value)}/{module.limit}">\n{module.value}\n</{section}>')
        return "\n".join(section_strs)

    old_repr_str = old_repr(sample_memory)
    new_repr_str = sample_memory.compile()
    assert new_repr_str == old_repr_str, f"Expected '{old_repr_str}' to be '{new_repr_str}'"


def test_memory_jinja2_set_template(sample_memory: Memory):
    """Test setting the template for the memory"""

    example_template = sample_memory.get_prompt_template()

    # Try setting a valid template
    sample_memory.set_prompt_template(prompt_template=example_template)

    # Try setting an invalid template (bad jinja2)
    template_bad_jinja = (
        "{% for section, module in mammoth.items() %}"
        '<{{ section }} characters="{{ module.value|length }}/{{ module.limit }}">\n'
        "{{ module.value }}\n"
        "</{{ section }}>"
        "{% if not loop.last %}\n{% endif %}"
        "{% endfor %"  # Missing closing curly brace
    )
    with pytest.raises(ValueError):
        sample_memory.set_prompt_template(prompt_template=template_bad_jinja)

    # Try setting an invalid template (not compatible with memory structure)
    template_bad_memory_structure = (
        "{% for section, module in mammoth.items() %}"
        '<{{ section }} characters="{{ module.value|length }}/{{ module.limit }}">\n'
        "{{ module.value }}\n"
        "</{{ section }}>"
        "{% if not loop.last %}\n{% endif %}"
        "{% endfor %}"
    )
    with pytest.raises(ValueError):
        sample_memory.set_prompt_template(prompt_template=template_bad_memory_structure)
