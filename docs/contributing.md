# Contributing

## Installing from source

To install MemGPT from source, start by cloning the repo:
```sh
git clone git@github.com:cpacker/MemGPT.git
```

**Installing dependencies with poetry** (recommended):

First, install Poetry using [the official instructions here](https://python-poetry.org/docs/#installing-with-the-official-installer).

```shell
cd MemGPT
pip install poetry
poetry install -E dev -E postgres -E local -E legacy
```
If you are managing dependencies with poetry, you will need to run MemGPT commands with `poetry run memgpt run`. 

**Installing dependencies with pip**:
```shell
cd MemGPT
# Optional: set up a virtual environment.
# python3 -m venv venv
# . venv/bin/activate
pip install -e '.[dev,postgres,local,legacy]'
```

Now, you should be able to run `memgpt` from the command-line using the downloaded source code.

If you are having dependency issues using `pip install -e .`, we recommend you install the package using Poetry (see below). Installing MemGPT from source using Poetry will ensure that you are using exact package versions that have been tested for the production build.

### Contributing to the MemGPT project

We welcome pull requests! Please see [our contributing guide](https://github.com/cpacker/MemGPT/blob/main/CONTRIBUTING.md) for instructions on how to contribute to the project.
