## Installing from source

To install MemGPT from source, start by cloning the repo:
```sh
git clone git@github.com:cpacker/MemGPT.git
```

### Installing dependencies with poetry (recommended)

First, install Poetry using [the official instructions here](https://python-poetry.org/docs/#installation).

Once Poetry is installed, navigate to the MemGPT directory and install the MemGPT project with Poetry:
```shell
cd MemGPT
poetry shell
poetry install -E dev -E postgres -E local 
```

Now when you want to use `memgpt`, make sure you first activate the `poetry` environment using poetry shell:
```shell
$ poetry shell
(pymemgpt-py3.10) $ memgpt run
```

Alternatively, you can use `poetry run` (which will activate the `poetry` environment for the `memgpt run` command only):
```shell
poetry run memgpt run
```

### Installing dependencies with pip

First you should set up a dedicated virtual environment. This is optional, but is highly recommended:
```shell
cd MemGPT
python3 -m venv venv
. venv/bin/activate
```

Once you've activated your virtual environment and are in the MemGPT project directory, you can install the dependencies with `pip`:
```shell
pip install -e '.[dev,postgres,local]'
```

Now, you should be able to run `memgpt` from the command-line using the downloaded source code (if you used a virtual environment, you have to activate the virtual environment to access `memgpt`):
```shell
$ . venv/bin/activate
(venv) $ memgpt run
```

If you are having dependency issues using `pip`, we recommend you install the package using Poetry. Installing MemGPT from source using Poetry will ensure that you are using exact package versions that have been tested for the production build.

## Contributing to the MemGPT project

We welcome pull requests! Please see [our contributing guide](https://github.com/cpacker/MemGPT/blob/main/CONTRIBUTING.md) for instructions on how to contribute to the project.
