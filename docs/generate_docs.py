import os

from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.smart import SmartProcessor
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer


def generate_config(package):
    config = PydocMarkdown(
        loaders=[PythonLoader(packages=[package])],
        processors=[FilterProcessor(skip_empty_modules=True), CrossrefProcessor(), SmartProcessor()],
        renderer=MarkdownRenderer(
            render_module_header=False,
            descriptive_class_title=False,
        ),
    )
    return config


def generate_modules(config):
    modules = config.load_modules()
    config.process(modules)
    return modules


# get PYTHON_DOC_DIR from environment
folder = os.getenv("PYTHON_DOC_DIR")
assert folder is not None, "PYTHON_DOC_DIR environment variable must be set"


# Generate client documentation. This takes the documentation from the AbstractClient, but then appends the documentation from the LocalClient and RESTClient.
config = generate_config("letta.client")
modules = generate_modules(config)

## Get members from AbstractClient
##for module in generate_modules(config):
# for module in modules:
#    client_members = [m for m in module.members if m.name == "AbstractClient"]
#    if len(client_members) > 0:
#        break
#
# client_members = client_members[0].members
# print(client_members)

# Add members and render for LocalClient and RESTClient
# config = generate_config("letta.client")

for module_name in ["LocalClient", "RESTClient"]:
    for module in generate_modules(config):
        # for module in modules:
        members = [m for m in module.members if m.name == module_name]
        if len(members) > 0:
            print(module_name)
            # module.members = members + client_members
            # print(module_name, members)
            module.members = members
            open(os.path.join(folder, f"{module_name}.mdx"), "w").write(config.renderer.render_to_string([module]))
            break


# Documentation of schemas
schema_config = generate_config("letta.schemas")

schema_models = [
    "LettaBase",
    "LettaConfig",
    "Message",
    "Passage",
    "AgentState",
    "File",
    "Source",
    "LLMConfig",
    "EmbeddingConfig",
    "LettaRequest",
    "LettaResponse",
    ["LettaMessage", "FunctionCallMessage", "FunctionReturn", "InternalMonologue"],
    "LettaUsageStatistics",
    ["Memory", "BasicBlockMemory", "ChatMemory"],
    "Block",
    # ["Job", "JobStatus"],
    "Job",
    "Tool",
    "User",
]
for module_name in schema_models:
    for module in generate_modules(schema_config):
        if isinstance(module_name, list):
            # multiple objects in the same file
            members = [m for m in module.members if m.name in module_name]
            title = module_name[0]
        else:
            # single object in a file
            members = [m for m in module.members if m.name == module_name]
            title = module_name
        if len(members) > 0:
            print(module_name)
            module.members = members
            open(os.path.join(folder, f"{title}.mdx"), "w").write(config.renderer.render_to_string([module]))
            break

# Documentation for connectors
connectors = ["DataConnector", "DirectoryConnector"]
connector_config = generate_config("letta.data_sources")
for module_name in connectors:
    for module in generate_modules(connector_config):
        members = [m for m in module.members if m.name == module_name]
        if len(members) > 0:
            print(module_name)
            module.members = members
            open(os.path.join(folder, f"{module_name}.mdx"), "w").write(config.renderer.render_to_string([module]))
            break


## TODO: append the rendering from LocalClient and RESTClient from AbstractClient
#
## TODO: add documentation of schemas
#
# for module in modules:
#    print(module.name, type(module))
#    print(module)
#
#    #module_name = "AbstractClient"
#    #members = [m for m in module.members if m.name == module_name]
#    #print([m.name for m in members])
#    #module.members = members
#
#    if "__" in module.name:
#        continue
#    #if len(members) > 0:
#    #    open(os.path.join(folder, f"{module_name}.md"), "w").write(config.renderer.render_to_string([module]))
#    open(os.path.join(folder, f"{module.name}.md"), "w").write(config.renderer.render_to_string([module]))
