from typing import Any, Optional, Union

import humps
from pydantic import BaseModel


def generate_composio_tool_wrapper(action: "ActionType") -> tuple[str, str]:
    # Instantiate the object
    tool_instantiation_str = f"composio_toolset.get_tools(actions=[Action.{str(action)}])[0]"

    # Generate func name
    func_name = action.name.lower()

    wrapper_function_str = f"""
def {func_name}(**kwargs):
    if 'self' in kwargs:
        del kwargs['self']
    from composio import Action, App, Tag
    from composio_langchain import ComposioToolSet

    composio_toolset = ComposioToolSet()
    tool = {tool_instantiation_str}
    return tool.func(**kwargs)['data']
    """

    # Compile safety check
    assert_code_gen_compilable(wrapper_function_str)

    return func_name, wrapper_function_str


def generate_langchain_tool_wrapper(
    tool: "LangChainBaseTool", additional_imports_module_attr_map: dict[str, str] = None
) -> tuple[str, str]:
    tool_name = tool.__class__.__name__
    import_statement = f"from langchain_community.tools import {tool_name}"
    extra_module_imports = generate_import_code(additional_imports_module_attr_map)

    # Safety check that user has passed in all required imports:
    assert_all_classes_are_imported(tool, additional_imports_module_attr_map)

    tool_instantiation = f"tool = {generate_imported_tool_instantiation_call_str(tool)}"
    run_call = f"return tool._run(**kwargs)"
    func_name = humps.decamelize(tool_name)

    # Combine all parts into the wrapper function
    wrapper_function_str = f"""
def {func_name}(**kwargs):
    if 'self' in kwargs:
        del kwargs['self']
    import importlib
    {import_statement}
    {extra_module_imports}
    {tool_instantiation}
    {run_call}
"""

    # Compile safety check
    assert_code_gen_compilable(wrapper_function_str)

    return func_name, wrapper_function_str


def generate_crewai_tool_wrapper(tool: "CrewAIBaseTool", additional_imports_module_attr_map: dict[str, str] = None) -> tuple[str, str]:
    tool_name = tool.__class__.__name__
    import_statement = f"from crewai_tools import {tool_name}"
    extra_module_imports = generate_import_code(additional_imports_module_attr_map)

    # Safety check that user has passed in all required imports:
    assert_all_classes_are_imported(tool, additional_imports_module_attr_map)

    tool_instantiation = f"tool = {generate_imported_tool_instantiation_call_str(tool)}"
    run_call = f"return tool._run(**kwargs)"
    func_name = humps.decamelize(tool_name)

    # Combine all parts into the wrapper function
    wrapper_function_str = f"""
def {func_name}(**kwargs):
    if 'self' in kwargs:
        del kwargs['self']
    import importlib
    {import_statement}
    {extra_module_imports}
    {tool_instantiation}
    {run_call}
"""

    # Compile safety check
    assert_code_gen_compilable(wrapper_function_str)

    return func_name, wrapper_function_str


def assert_code_gen_compilable(code_str):
    try:
        compile(code_str, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax error in code: {e}")


def assert_all_classes_are_imported(
    tool: Union["LangChainBaseTool", "CrewAIBaseTool"], additional_imports_module_attr_map: dict[str, str]
) -> None:
    # Safety check that user has passed in all required imports:
    tool_name = tool.__class__.__name__
    current_class_imports = {tool_name}
    if additional_imports_module_attr_map:
        current_class_imports.update(set(additional_imports_module_attr_map.values()))
    required_class_imports = set(find_required_class_names_for_import(tool))

    if not current_class_imports.issuperset(required_class_imports):
        err_msg = f"[ERROR] You are missing module_attr pairs in `additional_imports_module_attr_map`. Currently, you have imports for {current_class_imports}, but the required classes for import are {required_class_imports}"
        print(err_msg)
        raise RuntimeError(err_msg)


def find_required_class_names_for_import(obj: Union["LangChainBaseTool", "CrewAIBaseTool", BaseModel]) -> list[str]:
    """
    Finds all the class names for required imports when instantiating the `obj`.
    NOTE: This does not return the full import path, only the class name.

    We accomplish this by running BFS and deep searching all the BaseModel objects in the obj parameters.
    """
    class_names = {obj.__class__.__name__}
    queue = [obj]

    while queue:
        # Get the current object we are inspecting
        curr_obj = queue.pop()

        # Collect all possible candidates for BaseModel objects
        candidates = []
        if is_base_model(curr_obj):
            # If it is a base model, we get all the values of the object parameters
            # i.e., if obj('b' = <class A>), we would want to inspect <class A>
            fields = dict(curr_obj)
            # Generate code for each field, skipping empty or None values
            candidates = list(fields.values())
        elif isinstance(curr_obj, dict):
            # If it is a dictionary, we get all the values
            # i.e., if obj = {'a': 3, 'b': <class A>}, we would want to inspect <class A>
            candidates = list(curr_obj.values())
        elif isinstance(curr_obj, list):
            # If it is a list, we inspect all the items in the list
            # i.e., if obj = ['a', 3, None, <class A>], we would want to inspect <class A>
            candidates = curr_obj

        # Filter out all candidates that are not BaseModels
        # In the list example above, ['a', 3, None, <class A>], we want to filter out 'a', 3, and None
        candidates = filter(lambda x: is_base_model(x), candidates)

        # Classic BFS here
        for c in candidates:
            c_name = c.__class__.__name__
            if c_name not in class_names:
                class_names.add(c_name)
                queue.append(c)

    return list(class_names)


def generate_imported_tool_instantiation_call_str(obj: Any) -> Optional[str]:
    if isinstance(obj, (int, float, str, bool, type(None))):
        # This is the base case
        # If it is a basic Python type, we trivially return the string version of that value
        # Handle basic types
        return repr(obj)
    elif is_base_model(obj):
        # Otherwise, if it is a BaseModel
        # We want to pull out all the parameters, and reformat them into strings
        # e.g. {arg}={value}
        # The reason why this is recursive, is because the value can be another BaseModel that we need to stringify
        model_name = obj.__class__.__name__
        fields = obj.dict()
        # Generate code for each field, skipping empty or None values
        field_assignments = []
        for arg, value in fields.items():
            python_string = generate_imported_tool_instantiation_call_str(value)
            if python_string:
                field_assignments.append(f"{arg}={python_string}")

        assignments = ", ".join(field_assignments)
        return f"{model_name}({assignments})"
    elif isinstance(obj, dict):
        # Inspect each of the items in the dict and stringify them
        # This is important because the dictionary may contain other BaseModels
        dict_items = []
        for k, v in obj.items():
            python_string = generate_imported_tool_instantiation_call_str(v)
            if python_string:
                dict_items.append(f"{repr(k)}: {python_string}")

        joined_items = ", ".join(dict_items)
        return f"{{{joined_items}}}"
    elif isinstance(obj, list):
        # Inspect each of the items in the list and stringify them
        # This is important because the list may contain other BaseModels
        list_items = [generate_imported_tool_instantiation_call_str(v) for v in obj]
        filtered_list_items = list(filter(None, list_items))
        list_items = ", ".join(filtered_list_items)
        return f"[{list_items}]"
    else:
        # Otherwise, if it is none of the above, that usually means it is a custom Python class that is NOT a BaseModel
        # Thus, we cannot get enough information about it to stringify it
        # This may cause issues, but we are making the assumption that any of these custom Python types are handled correctly by the parent library, such as LangChain or CrewAI
        # An example would be that WikipediaAPIWrapper has an argument that is a wikipedia (pip install wikipedia) object
        # We cannot stringify this easily, but WikipediaAPIWrapper handles the setting of this parameter internally
        # This assumption seems fair to me, since usually they are external imports, and LangChain and CrewAI should be bundling those as module-level imports within the tool
        # We throw a warning here anyway and provide the class name
        print(
            f"[WARNING] Skipping parsing unknown class {obj.__class__.__name__} (does not inherit from the Pydantic BaseModel and is not a basic Python type)"
        )
        if obj.__class__.__name__ == "function":
            import inspect

            print(inspect.getsource(obj))

        return None


def is_base_model(obj: Any):
    from crewai_tools.tools.base_tool import BaseModel as CrewAiBaseModel
    from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel

    return isinstance(obj, BaseModel) or isinstance(obj, LangChainBaseModel) or isinstance(obj, CrewAiBaseModel)


def generate_import_code(module_attr_map: Optional[dict]):
    if not module_attr_map:
        return ""

    code_lines = []
    for module, attr in module_attr_map.items():
        module_name = module.split(".")[-1]
        code_lines.append(f"# Load the module\n    {module_name} = importlib.import_module('{module}')")
        code_lines.append(f"    # Access the {attr} from the module")
        code_lines.append(f"    {attr} = getattr({module_name}, '{attr}')")
    return "\n".join(code_lines)
