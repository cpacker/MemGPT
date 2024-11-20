# Tool Sandbox Environment (`tool_sandbox_env`)

The `tool_sandbox_env` is a local directory designed to serve as a secure and isolated sandbox environment for executing tools. It provides a virtualized workspace where each tool can operate independently, ensuring minimal interference with the host system or other tools.

## Key Features

- **Isolation**: Each sandbox operates within its own virtual environment (`venv`), ensuring that tool-specific dependencies do not conflict with global or system-level packages.
- **Custom Configuration**: Tools can access environment variables and configurations specific to their execution context.
