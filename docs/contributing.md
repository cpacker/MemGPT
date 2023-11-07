# Contributing

## Installing from source
First, install Poetry using [the official instructions here](https://python-poetry.org/docs/#installing-with-the-official-installer).

Then, you can install MemGPT from source with:
```sh
git clone git@github.com:cpacker/MemGPT.git
poetry shell
poetry install
```
We recommend installing pre-commit to ensure proper formatting during development:
```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Formatting
We welcome pull requests! Please run the formatter before submitting a pull request:
```sh
poetry run black . -l 140
```
