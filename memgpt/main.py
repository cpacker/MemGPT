import asyncio
import logging
import glob
import os
import sys
import pickle
import asyncio
import websockets
import typer
import threading
import time
import subprocess
import sys

import questionary
import typer

from rich.console import Console

console = Console()

import memgpt.interface  # for printing to terminal
import memgpt.agent as agent
import memgpt.system as system
import memgpt.utils as utils
import memgpt.presets as presets
import memgpt.constants as constants
import memgpt.personas.personas as personas
import memgpt.humans.humans as humans
from memgpt.persistence_manager import (
    InMemoryStateManager,
    InMemoryStateManagerWithPreloadedArchivalMemory,
    InMemoryStateManagerWithFaiss,
)

from memgpt.config import Config
from memgpt.constants import MEMGPT_DIR
import asyncio

app = typer.Typer()

####
AllowExternalInput = False

if AllowExternalInput:

    def check_server(uri, retry_interval=1):
        """Keep attempting to connect to the server at the specified interval. Set an event flag once successful."""
        async def attempt_connection():
            while True:
                try:
                    async with websockets.connect(uri):
                        # Connection successful
                        return True
                except (OSError, websockets.WebSocketException) as e:
                    # Connection failed
                    print(f"Connection attempt failed: {e}. Retrying in {retry_interval} seconds...")
                    await asyncio.sleep(retry_interval)

        # Run the connection attempt in the default event loop
        connected = asyncio.get_event_loop().run_until_complete(attempt_connection())

        if connected:
            print("Server is accepting connections.")
            server_ready_event.set()




    # Global event used to indicate that the server is ready
    server_ready_event = threading.Event()

    def run_server_script():
        # Start the server (assuming server.py is your server script)
        subprocess.Popen([sys.executable, './memgpt/server.py'],
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)

        # Here, you'd ideally have a way to ensure the server is indeed ready.
        # Since we're simulating, we just wait for a few seconds (e.g., the server usually takes a few seconds to be ready).
        print("Starting server...")
        # Simulating server start-up delay with sleep
        # threading.Event().wait(10)  # Assume it takes 10 seconds for the server to be ready
        server_ready_event.wait(10) 

        # Signal that the server is ready
        server_ready_event.set()

    def start_server():
        server_thread = threading.Thread(target=run_server_script, daemon=True)
        server_thread.start()

    # Your WebSocket client setup remains unchanged
    # ...

    # Function to start the WebSocket client connection
    def start_client():
        # Wait for the server to be ready
        print("Waiting for server to be ready...")
        server_is_ready = check_server("ws://localhost:1234")  # or whatever your server's address and port are

        if server_is_ready:
            print("Server is ready.")
            server_ready_event.set()
        else:
            print("Server failed to start.")
        
        
        # server_ready_event.wait()
        # print("Server is ready.")

        # URI of the WebSocket server to connect to
    uri = "ws://localhost:1234"
    # ws_client = WebSocketClient("ws ://localhost:1234")

    async def websocket_handler(uri, queue):
        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    # Wait for any message from the server and add it to the queue
                    message = await websocket.recv()
                    queue.put_nowait(message)
                except websockets.exceptions.ConnectionClosed:
                    print("Connection with server closed")
                    break

    class WebSocketClient:
        def __init__(self, uri):
            self.uri = uri
            self.loop = asyncio.new_event_loop()
            self.websocket = None
            self.incoming_messages = Queue()  # This queue will hold the incoming messages.

        async def connect(self):
            self.websocket = await websockets.connect(self.uri)
            await self.receive_messages()

        async def receive_messages(self):
            # This coroutine will handle incoming messages and put them in the queue.
            async for message in self.websocket:
                self.incoming_messages.put(message)

        def start(self):
            self.loop.run_until_complete(self.connect())


    def start_websocket_loop(loop, uri, queue):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(websocket_handler(uri, queue))

    # You can then have these as separate calls wherever needed, not necessarily in 'main'
    start_server()  # This would be called wherever you're starting your server from
    start_client()  # This would be called wherever you're starting your client from

#####

# AllowAutogenInput = True

# def run_server_script():
#     server_process = subprocess.Popen(
#         [sys.executable, './memgpt/server.py'],
#         stdout=subprocess.PIPE, 
#         stderr=subprocess.PIPE,
#         text=True  # to get the output in string format
#     )

#     # Communicate() is a blocking call. By putting it in a thread, 
#     # it won't block the main script.
#     stdout, stderr = server_process.communicate()
#     print(stdout)
#     if stderr:
#         print("Error in server script:", stderr)

# # If your condition for running the server script is True, start a thread.
# if AllowAutogenInput:
#     server_thread = threading.Thread(target=run_server_script, daemon=True)
#     server_thread.start()

#     print("Server started... continuing with other tasks.")

#########

# if AllowAutogenInput:
#     # Start the server without blocking the rest of your script.
#     # server_process = subprocess.Popen(
#     #     [sys.executable, './memgpt/server.py'],
#     #     # These stream arguments prevent the parent process from waiting on the subprocess.
#     #     stdout=subprocess.DEVNULL,  # You might want to redirect to a file instead.
#     #     stderr=subprocess.DEVNULL,
#     # )

#     print("Server started... continuing with other global level tasks.")

    
# else:
#     print("Variable is falsy (it evaluates to False).")

####
# # URI of the WebSocket server to connect to
# uri = "ws://localhost:1234"
# # ws_client = WebSocketClient("ws ://localhost:1234")

# async def websocket_handler(uri, queue):
#     async with websockets.connect(uri) as websocket:
#         while True:
#             try:
#                 # Wait for any message from the server and add it to the queue
#                 message = await websocket.recv()
#                 queue.put_nowait(message)
#             except websockets.exceptions.ConnectionClosed:
#                 print("Connection with server closed")
#                 break

# class WebSocketClient:
#     def __init__(self, uri):
#         self.uri = uri
#         self.loop = asyncio.new_event_loop()
#         self.websocket = None
#         self.incoming_messages = Queue()  # This queue will hold the incoming messages.

#     async def connect(self):
#         self.websocket = await websockets.connect(self.uri)
#         await self.receive_messages()

#     async def receive_messages(self):
#         # This coroutine will handle incoming messages and put them in the queue.
#         async for message in self.websocket:
#             self.incoming_messages.put(message)

#     def start(self):
#         self.loop.run_until_complete(self.connect())


# def start_websocket_loop(loop, uri, queue):
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(websocket_handler(uri, queue))

####
        
def clear_line():
    if os.name == "nt":  # for windows
        console.print("\033[A\033[K", end="")
    else:  # for linux
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()


def save(memgpt_agent, cfg):
    filename = utils.get_local_time().replace(" ", "_").replace(":", "_")
    filename = f"{filename}.json"
    directory = os.path.join(MEMGPT_DIR, "saved_state")
    filename = os.path.join(directory, filename)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        memgpt_agent.save_to_json_file(filename)
        print(f"Saved checkpoint to: {filename}")
        cfg.agent_save_file = filename
    except Exception as e:
        print(f"Saving state to {filename} failed with: {e}")

    # save the persistence manager too
    filename = filename.replace(".json", ".persistence.pickle")
    try:
        memgpt_agent.persistence_manager.save(filename)
        print(f"Saved persistence manager to: {filename}")
        cfg.persistence_manager_save_file = filename
    except Exception as e:
        print(f"Saving persistence manager to {filename} failed with: {e}")
    cfg.write_config()


def load(memgpt_agent, filename):
    if filename is not None:
        if filename[-5:] != ".json":
            filename += ".json"
        try:
            memgpt_agent.load_from_json_file_inplace(filename)
            print(f"Loaded checkpoint {filename}")
        except Exception as e:
            print(f"Loading {filename} failed with: {e}")
    else:
        # Load the latest file
        print(
            f"/load warning: no checkpoint specified, loading most recent checkpoint instead"
        )
        json_files = glob.glob(
            "saved_state/*.json"
        )  # This will list all .json files in the current directory.

        # Check if there are any json files.
        if not json_files:
            print(f"/load error: no .json checkpoint files found")
        else:
            # Sort files based on modified timestamp, with the latest file being the first.
            filename = max(json_files, key=os.path.getmtime)
            try:
                memgpt_agent.load_from_json_file_inplace(filename)
                print(f"Loaded checkpoint {filename}")
            except Exception as e:
                print(f"Loading {filename} failed with: {e}")

    # need to load persistence manager too
    filename = filename.replace(".json", ".persistence.pickle")
    try:
        memgpt_agent.persistence_manager = InMemoryStateManager.load(
            filename
        )  # TODO(fixme):for different types of persistence managers that require different load/save methods
        print(f"Loaded persistence manager from {filename}")
    except Exception as e:
        print(
            f"/load warning: loading persistence manager from {filename} failed with: {e}"
        )


@app.command()
def run(
    persona: str = typer.Option(None, help="Specify persona"),
    human: str = typer.Option(None, help="Specify human"),
    model: str = typer.Option(
        constants.DEFAULT_MEMGPT_MODEL, help="Specify the LLM model"
    ),
    first: bool = typer.Option(
        False, "--first", help="Use --first to send the first message in the sequence"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Use --debug to enable debugging output"
    ),
    no_verify: bool = typer.Option(
        False, "--no_verify", help="Bypass message verification"
    ),
    archival_storage_faiss_path: str = typer.Option(
        "",
        "--archival_storage_faiss_path",
        help="Specify archival storage with FAISS index to load (a folder with a .index and .json describing documents to be loaded)",
    ),
    archival_storage_files: str = typer.Option(
        "",
        "--archival_storage_files",
        help="Specify files to pre-load into archival memory (glob pattern)",
    ),
    archival_storage_files_compute_embeddings: str = typer.Option(
        "",
        "--archival_storage_files_compute_embeddings",
        help="Specify files to pre-load into archival memory (glob pattern), and compute embeddings over them",
    ),
    archival_storage_sqldb: str = typer.Option(
        "",
        "--archival_storage_sqldb",
        help="Specify SQL database to pre-load into archival memory",
    ),
    use_azure_openai: bool = typer.Option(
        False,
        "--use_azure_openai",
        help="Use Azure OpenAI (requires additional environment variables)",
    ),  # TODO: just pass in?
):
    loop = asyncio.get_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        main(
            persona,
            human,
            model,
            first,
            debug,
            no_verify,
            archival_storage_faiss_path,
            archival_storage_files,
            archival_storage_files_compute_embeddings,
            archival_storage_sqldb,
            use_azure_openai,
        )
    )


async def main(
    persona,
    human,
    model,
    first,
    debug,
    no_verify,
    archival_storage_faiss_path,
    archival_storage_files,
    archival_storage_files_compute_embeddings,
    archival_storage_sqldb,
    use_azure_openai,
):
    
    message_queue = asyncio.Queue()
       # Start a new thread and event loop for the WebSocket client
    websocket_loop = asyncio.new_event_loop()
    websocket_thread = threading.Thread(target=start_websocket_loop, args=(websocket_loop, uri, message_queue), daemon=True)
    websocket_thread.start()

    try:
        while True:  # Your application's main loop
            if not message_queue.empty():
                message = message_queue.get_nowait()
                print(f"Processing received message: {message}")
            else:
                print("waiting")

            # ... the rest of your main loop code ...
            # This is a placeholder; your actual main loop logic will vary
            utils.DEBUG = debug
            logging.getLogger().setLevel(logging.CRITICAL)
            if debug:
                logging.getLogger().setLevel(logging.DEBUG)

        

            if any(
                (
                    persona,
                    human,
                    model != constants.DEFAULT_MEMGPT_MODEL,
                    archival_storage_faiss_path,
                    archival_storage_files,
                    archival_storage_files_compute_embeddings,
                    archival_storage_sqldb,
                )
            ):
                memgpt.interface.important_message("⚙️ Using legacy command line arguments.")
                model = model
                if model is None:
                    model = constants.DEFAULT_MEMGPT_MODEL
                memgpt_persona = persona
                if memgpt_persona is None:
                    memgpt_persona = (
                        personas.GPT35_DEFAULT if "gpt-3.5" in model else personas.DEFAULT,
                        Config.personas_dir,
                    )
                else:
                    try:
                        personas.get_persona_text(memgpt_persona, Config.custom_personas_dir)
                        memgpt_persona = (memgpt_persona, Config.custom_personas_dir)
                    except FileNotFoundError:
                        personas.get_persona_text(memgpt_persona, Config.personas_dir)
                        memgpt_persona = (memgpt_persona, Config.personas_dir)

                human_persona = human
                if human_persona is None:
                    human_persona = (humans.DEFAULT, Config.humans_dir)
                else:
                    try:
                        humans.get_human_text(human_persona, Config.custom_humans_dir)
                        human_persona = (human_persona, Config.custom_humans_dir)
                    except FileNotFoundError:
                        humans.get_human_text(human_persona, Config.humans_dir)
                        human_persona = (human_persona, Config.humans_dir)

                print(persona, model, memgpt_persona)
                if archival_storage_files:
                    cfg = await Config.legacy_flags_init(
                        model,
                        memgpt_persona,
                        human_persona,
                        load_type="folder",
                        archival_storage_files=archival_storage_files,
                        compute_embeddings=False,
                    )
                elif archival_storage_faiss_path:
                    cfg = await Config.legacy_flags_init(
                        model,
                        memgpt_persona,
                        human_persona,
                        load_type="folder",
                        archival_storage_files=archival_storage_faiss_path,
                        archival_storage_index=archival_storage_faiss_path,
                        compute_embeddings=True,
                    )
                elif archival_storage_files_compute_embeddings:
                    print(model)
                    print(memgpt_persona)
                    print(human_persona)
                    cfg = await Config.legacy_flags_init(
                        model,
                        memgpt_persona,
                        human_persona,
                        load_type="folder",
                        archival_storage_files=archival_storage_files_compute_embeddings,
                        compute_embeddings=True,
                    )
                elif archival_storage_sqldb:
                    cfg = await Config.legacy_flags_init(
                        model,
                        memgpt_persona,
                        human_persona,
                        load_type="sql",
                        archival_storage_files=archival_storage_sqldb,
                        compute_embeddings=False,
                    )
                else:
                    cfg = await Config.legacy_flags_init(
                        model,
                        memgpt_persona,
                        human_persona,
                    )
            else:
                cfg = await Config.config_init()

            memgpt.interface.important_message(
                "Running... [exit by typing '/exit', list available commands with '/help']"
            )
            if cfg.model != constants.DEFAULT_MEMGPT_MODEL:
                memgpt.interface.warning_message(
                    f"⛔️ Warning - you are running MemGPT with {cfg.model}, which is not officially supported (yet). Expect bugs!"
                )

            # Azure OpenAI support
            if use_azure_openai:
                azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
                azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
                azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
                azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
                if None in [
                    azure_openai_key,
                    azure_openai_endpoint,
                    azure_openai_version,
                    azure_openai_deployment,
                ]:
                    print(
                        f"Error: missing Azure OpenAI environment variables. Please see README section on Azure."
                    )
                    return

                import openai

                openai.api_type = "azure"
                openai.api_key = azure_openai_key
                openai.api_base = azure_openai_endpoint
                openai.api_version = azure_openai_version
                # deployment gets passed into chatcompletion
            else:
                azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
                if azure_openai_deployment is not None:
                    print(
                        f"Error: AZURE_OPENAI_DEPLOYMENT should not be set if --use_azure_openai is False"
                    )
                    return

            if cfg.index:
                persistence_manager = InMemoryStateManagerWithFaiss(
                    cfg.index, cfg.archival_database
                )
            elif cfg.archival_storage_files:
                print(f"Preloaded {len(cfg.archival_database)} chunks into archival memory.")
                persistence_manager = InMemoryStateManagerWithPreloadedArchivalMemory(
                    cfg.archival_database
                )
            else:
                persistence_manager = InMemoryStateManager()

            if archival_storage_files_compute_embeddings:
                memgpt.interface.important_message(
                    f"(legacy) To avoid computing embeddings next time, replace --archival_storage_files_compute_embeddings={archival_storage_files_compute_embeddings} with\n\t --archival_storage_faiss_path={cfg.archival_storage_index} (if your files haven't changed)."
                )

            # Moved defaults out of FLAGS so that we can dynamically select the default persona based on model
            chosen_human = cfg.human_persona
            chosen_persona = cfg.memgpt_persona

            memgpt_agent = presets.use_preset(
                presets.DEFAULT,
                cfg.model,
                personas.get_persona_text(*chosen_persona),
                humans.get_human_text(*chosen_human),
                memgpt.interface,
                persistence_manager,
            )
            print_messages = memgpt.interface.print_messages
            print("printing message")
            await print_messages(memgpt_agent.messages)

            counter = 0
            user_input = None
            skip_next_user_input = False
            user_message = None
            USER_GOES_FIRST = first

            if cfg.load_type == "sql":  # TODO: move this into config.py in a clean manner
                if not os.path.exists(cfg.archival_storage_files):
                    print(f"File {cfg.archival_storage_files} does not exist")
                    return
                # Ingest data from file into archival storage
                else:
                    print(f"Database found! Loading database into archival memory")
                    data_list = utils.read_database_as_list(cfg.archival_storage_files)
                    user_message = f"Your archival memory has been loaded with a SQL database called {data_list[0]}, which contains schema {data_list[1]}. Remember to refer to this first while answering any user questions!"
                    for row in data_list:
                        await memgpt_agent.persistence_manager.archival_memory.insert(row)
                    print(f"Database loaded into archival memory.")

            if cfg.agent_save_file:
                load_save_file = await questionary.confirm(
                    f"Load in saved agent '{cfg.agent_save_file}'?"
                ).ask_async()
                if load_save_file:
                    load(memgpt_agent, cfg.agent_save_file)

            # auto-exit for
            if "GITHUB_ACTIONS" in os.environ:
                return

            if not USER_GOES_FIRST:
                console.input(
                    "[bold cyan]Hit enter to begin (will request first MemGPT message)[/bold cyan]"
                )
                clear_line()
                print()

            

            multiline_input = False
            while True:
                
                if not skip_next_user_input and (counter > 0 or USER_GOES_FIRST):
                    message = None
                    while not message:
                        if not message_queue.empty():
                            message = message_queue.get_nowait()
                            print(f"Processing received message: {message}")
                        else:
                            print("waiting")
                            time.sleep(1)  # Pause for a second to prevent busy-waiting.
                        print(message_queue)
                        # print("any news?")
                        if message:
                            print(message)

                    # Outside the loop, process the received message.
                    user_input = message if message else "This is default input. Do not respond."
                    

                # This has been commented out to allow Autogen to give input directly

                #      # Outside the loop, process the received message.
                #     user_input = message if message else "This is default input. Do not respond."
                #     # Ask for user input
                #     # user_input = console.input("[bold cyan]Enter your message:[/bold cyan] ")
                    
                #     # user_input = await questionary.text(
                #     #     "Enter your message:",
                #     #     multiline=multiline_input,
                    #     qmark=">",
                    # ).ask_async()
                    # clear_line()


                  


                    user_input = user_input.rstrip()

                    if user_input.startswith("!"):
                        print(f"Commands for CLI begin with '/' not '!'")
                        continue

                    if user_input == "":
                        # no empty messages allowed
                        print("Empty input received. Try again!")
                        continue

                    # Handle CLI commands
                    # Commands to not get passed as input to MemGPT
                    if user_input.startswith("/"):
                        if user_input.lower() == "/exit":
                            # autosave
                            save(memgpt_agent=memgpt_agent, cfg=cfg)
                            break

                        elif user_input.lower() == "/savechat":
                            filename = (
                                utils.get_local_time().replace(" ", "_").replace(":", "_")
                            )
                            filename = f"{filename}.pkl"
                            directory = os.path.join(MEMGPT_DIR, "saved_chats")
                            try:
                                if not os.path.exists(directory):
                                    os.makedirs(directory)
                                with open(os.path.join(directory, filename), "wb") as f:
                                    pickle.dump(memgpt_agent.messages, f)
                                    print(f"Saved messages to: {filename}")
                            except Exception as e:
                                print(f"Saving chat to {filename} failed with: {e}")
                            continue

                        elif user_input.lower() == "/save":
                            save(memgpt_agent=memgpt_agent, cfg=cfg)
                            continue

                        elif user_input.lower() == "/load" or user_input.lower().startswith(
                            "/load "
                        ):
                            command = user_input.strip().split()
                            filename = command[1] if len(command) > 1 else None
                            load(memgpt_agent=memgpt_agent, filename=filename)
                            continue

                        elif user_input.lower() == "/dump":
                            await print_messages(memgpt_agent.messages)
                            continue

                        elif user_input.lower() == "/dumpraw":
                            await memgpt.interface.print_messages_raw(memgpt_agent.messages)
                            continue

                        elif user_input.lower() == "/dump1":
                            await print_messages(memgpt_agent.messages[-1])
                            continue

                        elif user_input.lower() == "/memory":
                            print(f"\nDumping memory contents:\n")
                            print(f"{str(memgpt_agent.memory)}")
                            print(f"{str(memgpt_agent.persistence_manager.archival_memory)}")
                            print(f"{str(memgpt_agent.persistence_manager.recall_memory)}")
                            continue

                        elif user_input.lower() == "/model":
                            if memgpt_agent.model == "gpt-4":
                                memgpt_agent.model = "gpt-3.5-turbo"
                            elif memgpt_agent.model == "gpt-3.5-turbo":
                                memgpt_agent.model = "gpt-4"
                            print(f"Updated model to:\n{str(memgpt_agent.model)}")
                            continue

                        elif user_input.lower() == "/pop" or user_input.lower().startswith(
                            "/pop "
                        ):
                            # Check if there's an additional argument that's an integer
                            command = user_input.strip().split()
                            amount = (
                                int(command[1])
                                if len(command) > 1 and command[1].isdigit()
                                else 2
                            )
                            print(f"Popping last {amount} messages from stack")
                            for _ in range(min(amount, len(memgpt_agent.messages))):
                                memgpt_agent.messages.pop()
                            continue

                        # No skip options
                        elif user_input.lower() == "/wipe":
                            memgpt_agent = agent.AgentAsync(memgpt.interface)
                            user_message = None

                        elif user_input.lower() == "/heartbeat":
                            user_message = system.get_heartbeat()

                        elif user_input.lower() == "/memorywarning":
                            user_message = system.get_token_limit_warning()

                        elif user_input.lower() == "//":
                            multiline_input = not multiline_input
                            continue

                        elif user_input.lower() == "/" or user_input.lower() == "/help":
                            questionary.print("CLI commands", "bold")
                            for cmd, desc in USER_COMMANDS:
                                questionary.print(cmd, "bold")
                                questionary.print(f" {desc}")
                            continue

                        else:
                            print(f"Unrecognized command: {user_input}")
                            continue

                    else:
                        # If message did not begin with command prefix, pass inputs to MemGPT
                        # Handle user message and append to messages
                        user_message = system.package_user_message(user_input)

                skip_next_user_input = False
                print(user_message)

                with console.status("[bold cyan]Thinking...") as status:
                    (
                        new_messages,
                        heartbeat_request,
                        function_failed,
                        token_warning,
                    ) = await memgpt_agent.step(
                        user_message, first_message=False, skip_verify=no_verify
                    )

                    # Skip user inputs if there's a memory warning, function execution failed, or the agent asked for control
                    if token_warning:
                        user_message = system.get_token_limit_warning()
                        skip_next_user_input = True
                    elif function_failed:
                        user_message = system.get_heartbeat(
                            constants.FUNC_FAILED_HEARTBEAT_MESSAGE
                        )
                        skip_next_user_input = True
                    elif heartbeat_request:
                        user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                        skip_next_user_input = True

                counter += 1

            print("Finished.")
            print("Main function is running concurrently...")
            time.sleep(1)  # Simulating work by sleeping for 1 second

    except KeyboardInterrupt:
        print("Application closed by user.")
    finally:
        # Cleanup and close websocket connection
        websocket_loop.call_soon_threadsafe(websocket_loop.stop)
        websocket_thread.join()

    


USER_COMMANDS = [
    ("//", "toggle multiline input mode"),
    ("/exit", "exit the CLI"),
    ("/save", "save a checkpoint of the current agent/conversation state"),
    ("/load", "load a saved checkpoint"),
    ("/dump", "view the current message log (see the contents of main context)"),
    ("/memory", "print the current contents of agent memory"),
    ("/pop", "undo the last message in the conversation"),
    ("/heartbeat", "send a heartbeat system message to the agent"),
    ("/memorywarning", "send a memory warning system message to the agent"),
]


# stdout, stderr = server_process.communicate()
# if server_process.returncode != 0:
#     print(f"server.py ended with error code {server_process.returncode}.")
#     print(stderr.decode())
# else:
#     print(stdout.decode())