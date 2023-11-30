from memgpt.main import app
import typer

typer.secho(
    "Command `python main.py` no longer supported. Please run `memgpt run`. See https://memgpt.readthedocs.io/en/latest/quickstart/.",
    fg=typer.colors.YELLOW,
)
