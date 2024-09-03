# SyncServer

## Preamble
MemGPT is undergoing significant refactoring and migrating to a classic MVC pattern.The goal is to introduce software best practices to greatly increase reliability, reduce cycle times, and support rapid test driven development.

**Note:** This README should represent current state as it evolves and serve as a cheat sheet for developers during the process. Please keep this up to date as you evolve the code! Just like MemGPT agents manage their own core memory to preserve state, use this README as the shared developer "core memory" so we don't waste cycles on cognative load.

## Current State
[SyncServer](./server.py) behaves as a single monolith MVC Controller. On either side of the SyncServer (Controller) we have:
- Models are currently not managed via ORM. DB/memory syncing does **not** use sqlalchemy metadata - it is managed externally by a bespoke in-memory state manager called[PersistanceManger](../memgpt/persistence_manager.py). Models themselves are piped into one large interface class [MetadataStore](../metadata.py) that exposes methods for each CRUD action on all the Models. There is also quite a bit of Controller business logic stored in this MetadataStore object (and vice versa).
- Clients (views) are the CLI, python client and rest API. In some places they duplicate the Controller stand-up and configuration (TODO: should really expand on this once we dig into clients).
- Controller layer also owns the application configuration and startup (via a MemGPTConfig object that is parsed in the controller init).

- Configuration is via stack of envars, files, and a [constants module](../constants.py). The config is re-assessed in all 3 layers (and a few places outside the MVC stack). Each assessment adds a variety of defaults and modifiers - so we've got shared mutable state spiderwebbed all over the codebase in the form of configs.