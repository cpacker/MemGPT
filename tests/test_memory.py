import pytest

# Import the classes here, assuming the above definitions are in a module named memory_module
from memgpt.schemas.memory import ChatMemory, Memory


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
    memory_dict = sample_memory.to_dict()
    assert isinstance(memory_dict, dict)
    assert "persona" in memory_dict
    assert memory_dict["persona"]["value"] == "Chat Agent"


def test_load_memory_from_json(sample_memory: Memory):
    """Test loading ChatMemory from a JSON compatible dictionary"""
    memory_dict = sample_memory.to_dict()
    print(memory_dict)
    new_memory = Memory.load(memory_dict)
    assert new_memory.get_block("persona").value == "Chat Agent"
    assert new_memory.get_block("human").value == "User"


# def test_memory_functionality(sample_memory):
#    """Test memory modification functions"""
#    # Get memory functions
#    functions = get_memory_functions(ChatMemory)
#    # Test core_memory_append function
#    append_func = functions['core_memory_append']
#    print("FUNCTIONS", functions)
#    env = {}
#    env.update(globals())
#    for tool in functions:
#        # WARNING: name may not be consistent?
#        exec(tool.source_code, env)
#
#    print(exec)
#
#    append_func(sample_memory, 'persona', " is a test.")
#    assert sample_memory.memory['persona'].value == "Chat Agent\n is a test."
#    # Test core_memory_replace function
#    replace_func = functions['core_memory_replace']
#    replace_func(sample_memory, 'persona', " is a test.", " was a test.")
#    assert sample_memory.memory['persona'].value == "Chat Agent\n was a test."


def test_memory_limit_validation(sample_memory: Memory):
    """Test exceeding memory limit"""
    with pytest.raises(ValueError):
        ChatMemory(persona="x" * 3000, human="y" * 3000)

    with pytest.raises(ValueError):
        sample_memory.get_block("persona").value = "x" * 3000


def test_memory_jinja2_template(sample_memory: Memory):
    """Test to make sure the jinja2 template string is equivalent to the old __repr__ method"""

    def old_repr(self: Memory) -> str:
        """Generate a string representation of the memory in-context"""
        section_strs = []
        for section, module in self.memory.items():
            section_strs.append(f'<{section} characters="{len(module)}/{module.limit}">\n{module.value}\n</{section}>')
        return "\n".join(section_strs)

    old_repr_str = old_repr(sample_memory)
    new_repr_str = sample_memory.compile()
    assert new_repr_str == old_repr_str, f"Expected '{old_repr_str}' to be '{new_repr_str}'"
