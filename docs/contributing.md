# Contributing

## Installing from source

To install MemGPT from source, start by cloning the repo:
```sh
git clone git@github.com:cpacker/MemGPT.git
```

Then navigate to the main `MemGPT` directory, and do:
```sh
pip install -e .
```

Now, you should be able to run `memgpt` from the command-line using the downloaded source code.

If you are having dependency issues using `pip install -e .`, we recommend you install the package using Poetry (see below). Installing MemGPT from source using Poetry will ensure that you are using exact package versions that have been tested for the production build.

## Installing from source (using Poetry)

First, install Poetry using [the official instructions here](https://python-poetry.org/docs/#installing-with-the-official-installer).

Then, you can install MemGPT from source with:
```sh
git clone git@github.com:cpacker/MemGPT.git
poetry install -E dev
```

### Formatting

We welcome pull requests! Please run the formatter before submitting a pull request:
```sh
poetry run black . -l 140
```

We recommend installing pre-commit to ensure proper formatting during development:
```sh
poetry run pre-commit install
poetry run pre-commit run --all-files
```
