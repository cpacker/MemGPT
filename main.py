import asyncio
from absl import app, flags
import logging
import os
import sys
import pickle
import readline

from rich.console import Console

import interface, memgpt.agent as agent, memgpt.system as system, memgpt.utils as utils, memgpt.presets as presets, memgpt.constants as constants, memgpt.personas.personas as personas, memgpt.humans.humans as humans
from memgpt.persistence_manager import InMemoryStateManager, InMemoryStateManagerWithPreloadedArchivalMemory, InMemoryStateManagerWithFaiss

console = Console()
FLAGS = flags.FLAGS

flags.DEFINE_string("persona", default=personas.DEFAULT, required=False, help="Specify persona")
flags.DEFINE_string("human", default=humans.DEFAULT, required=False, help="Specify human")
flags.DEFINE_string("model", default=constants.DEFAULT_MEMGPT_MODEL, required=False, help="Specify the LLM model")
flags.DEFINE_boolean("first", default=False, required=False, help="Use -first to send the first message in the sequence")
flags.DEFINE_boolean("debug", default=False, required=False, help="Use -debug to enable debugging output")
flags.DEFINE_string("archival_storage_faiss_path", default="", required=False, help="Specify archival storage with FAISS index to load (a folder with a .index and .json describing documents to be loaded)")
flags.DEFINE_string("archival_storage_files", default="", required=False, help="Specify files to pre-load into archival memory (glob pattern)")
flags.DEFINE_string("archival_storage_files_compute_embeddings", default="", required=False, help="Specify files to pre-load into archival memory (glob pattern), and compute embeddings over them")
flags.DEFINE_string("archival_storage_sqldb", default="", required=False, help="Specify SQL database to pre-load into archival memory")

def clear_line():
    if os.name == 'nt':
        console.print("\033[A\033[K", end="")
    else:
        sys.stdout.write("\033[2K\033[G")
        sys.stdout.flush()

async def main():
    utils.DEBUG = FLAGS.debug
    logging.getLogger().setLevel(logging.CRITICAL)
    if FLAGS.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    print("Running... [exit by typing '/exit']")

    if FLAGS.archival_storage_faiss_path:
        index, archival_database = utils.prepare_archival_index(FLAGS.archival_storage_faiss_path)
        persistence_manager = InMemoryStateManagerWithFaiss(index, archival_database)
    elif FLAGS.archival_storage_files:
        archival_database = utils.prepare_archival_index_from_files(FLAGS.archival_storage_files)
        print(f"Preloaded {len(archival_database)} chunks into archival memory.")
        persistence_manager = InMemoryStateManagerWithPreloadedArchivalMemory(archival_database)
    elif FLAGS.archival_storage_files_compute_embeddings:
        faiss_save_dir = await utils.prepare_archival_index_from_files_compute_embeddings(FLAGS.archival_storage_files_compute_embeddings)
        interface.important_message(f"To avoid computing embeddings next time, replace --archival_storage_files_compute_embeddings={FLAGS.archival_storage_files_compute_embeddings} with\n\t --archival_storage_faiss_path={faiss_save_dir} (if your files haven't changed).")
        index, archival_database = utils.prepare_archival_index(faiss_save_dir)
        persistence_manager = InMemoryStateManagerWithFaiss(index, archival_database)
    else:
        persistence_manager = InMemoryStateManager()
    memgpt_agent = presets.use_preset(presets.DEFAULT, FLAGS.model, personas.get_persona_text(FLAGS.persona), humans.get_human_text(FLAGS.human), interface, persistence_manager)
    print_messages = interface.print_messages
    await print_messages(memgpt_agent.messages)

    counter = 0
    user_input = None
    skip_next_user_input = False
    user_message = None
    USER_GOES_FIRST = FLAGS.first

    if FLAGS.archival_storage_sqldb:
        if not os.path.exists(FLAGS.archival_storage_sqldb):
            print(f"File {user_input} does not exist")
            return
        else:
            print(f"Database found! Loading database into archival memory")
            data_list = utils.read_database_as_list(FLAGS.archival_storage_sqldb)
            user_message = f"Your archival memory has been loaded with a SQL database called {data_list[0]}, which contains schema {data_list[1]}. Remember to refer to this first while answering any user questions!"
            for row in data_list:
                await memgpt_agent.persistence_manager.archival_memory.insert(row)
            print(f"Database loaded into archival memory.")

    if not USER_GOES_FIRST:
        console.input('[bold cyan]Hit enter to begin (will request first MemGPT message)[/bold cyan]')
        clear_line()
        print()

    while True:
        if not skip_next_user_input and (counter > 0 or USER_GOES_FIRST):
            user_input = console.input("[bold cyan]Enter your message:[/bold cyan] ")
            clear_line()

            if user_input.startswith('/'):
                if user_input == "//":
                    print("Entering multiline mode, type // when done")
                    user_input_list = []
                    while True:
                        user_input = console.input("[bold cyan]>[/bold cyan] ")
                        clear_line()
                        if user_input == "//":
                            break
                        else:
                            user_input_list.append(user_input)
                    user_message = system.package_user_message("\n".join(user_input_list)
                # Rest of the command handling...
            else:
                user_message = system.package_user_message(user_input)
            # Rest of user input handling...

        # Rest of the processing...
    print("Finished.")

if __name__ == '__main__':
    def run(argv):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    app.run(run)
