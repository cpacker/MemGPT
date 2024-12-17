# 🚀 How to Contribute to Letta

Thank you for investing time in contributing to our project! Here's a guide to get you started.

## 1. 🚀 Getting Started

### 🍴 Fork the Repository

First things first, let's get you a personal copy of Letta to play with. Think of it as your very own playground. 🎪

1. Head over to the Letta repository on GitHub.
2. In the upper-right corner, hit the 'Fork' button.

### 🚀 Clone the Repository

Now, let's bring your new playground to your local machine.

```shell
git clone https://github.com/your-username/letta.git
```

### 🧩 Install Dependencies

First, install Poetry using [the official instructions here](https://python-poetry.org/docs/#installation).

Once Poetry is installed, navigate to the Letta directory and install the Letta project with Poetry:
```shell
cd Letta
poetry shell
poetry install --all-extras
```

Now when you want to use `letta`, make sure you first activate the `poetry` environment using poetry shell:
```shell
$ poetry shell
(pyletta-py3.12) $ letta run
```

Alternatively, you can use `poetry run` (which will activate the `poetry` environment for the `letta run` command only):
```shell
poetry run letta run
```

#### Installing pre-commit
We recommend installing pre-commit to ensure proper formatting during development:
```
poetry run pre-commit install
poetry run pre-commit run --all-files
```
If you don't install pre-commit, you will need to run `poetry run black .` before submitting a PR.

## 2. 🛠️ Making Changes

### 🌟 Create a Branch

Time to put on your creative hat and make some magic happen. First, let's create a new branch for your awesome changes. 🧙‍♂️

```shell
git checkout -b feature/your-feature
```

### ✏️ Make your Changes

Now, the world is your oyster! Go ahead and craft your fabulous changes. 🎨


#### Handling Database Migrations
If you are running Letta for the first time, your database will be automatically be setup. If you are updating Letta, you may need to run migrations. To run migrations, use the following command:
```shell
poetry run alembic upgrade head
```

#### Creating a new Database Migration
If you have made changes to the database models, you will need to create a new migration. To create a new migration, use the following command:
```shell
poetry run alembic revision --autogenerate -m "Your migration message here"
```

Visit the [Alembic documentation](https://alembic.sqlalchemy.org/en/latest/tutorial.html) for more information on creating and running migrations.

## 3. ✅ Testing

Before we hit the 'Wow, I'm Done' button, let's make sure everything works as expected. Run tests and make sure the existing ones don't throw a fit. And if needed, create new tests. 🕵️

### Run existing tests

Running tests if you installed via poetry:
```
poetry run pytest -s tests
```

Running tests if you installed via pip:
```
pytest -s tests
```

### Creating new tests
If you added a major feature change, please add new tests in the `tests/` directory.

## 4. 🧩 Adding new dependencies
If you need to add a new dependency to Letta, please add the package via `poetry add <PACKAGE_NAME>`. This will update the `pyproject.toml` and `poetry.lock` files. If the dependency does not need to be installed by all users, make sure to mark the dependency as optional in the `pyproject.toml` file and if needed, create a new extra under `[tool.poetry.extras]`.

## 5. 🚀 Submitting Changes

### Check Formatting
Please ensure your code is formatted correctly by running:
```
poetry run black . -l 140
```

### 🚀 Create a Pull Request

You're almost there! It's time to share your brilliance with the world. 🌍

1. Visit [Letta](https://github.com/letta-ai/letta).
2. Click "New Pull Request" button.
3. Choose the base branch (`main`) and the compare branch (your feature branch).
4. Whip up a catchy title and describe your changes in the description. 🪄

## 6. 🔍 Review and Approval

The maintainers will take a look and might suggest some cool upgrades or ask for more details. Once they give the thumbs up, your creation becomes part of Letta!

## 7. 📜 Code of Conduct

Please be sure to follow the project's Code of Conduct.

## 8. 📫 Contact

Need help or just want to say hi? We're here for you. Reach out through filing an issue on this GitHub repository or message us on our [Discord server](https://discord.gg/9GEQrxmVyE).

Thanks for making Letta even more fantastic!

## WIP - 🐋 Docker Development
If you prefer to keep your resources isolated by developing purely in containers, you can start Letta in development with:
```shell
docker compose -f compose.yaml -f development.compose.yml up
```
This will volume mount your local codebase and reload the server on file changes.
