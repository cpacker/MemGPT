import os
import runpy


def generate_and_execute_tool(tool_name: str, args: dict):
    # Define the tool's directory and file
    tools_dir = os.path.join(os.path.dirname(__file__), "tools")
    script_path = os.path.join(tools_dir, f"{tool_name}_execution.py")

    # Generate the Python script
    with open(script_path, "w") as script_file:
        script_file.write(f"from restaurant_management_system.tools.{tool_name} import {tool_name}\n\n")
        arg_str = ", ".join([f"{key}={repr(value)}" for key, value in args.items()])
        script_file.write(f"if __name__ == '__main__':\n")
        script_file.write(f"    result = {tool_name}({arg_str})\n")
        script_file.write(f"    print(result)\n")

    # Execute the script
    runpy.run_path(script_path, run_name="__main__")

    # Optional: Clean up generated script
    # os.remove(script_path)


generate_and_execute_tool("adjust_menu_prices", {"percentage": 10})
