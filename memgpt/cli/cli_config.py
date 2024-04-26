import builtins
import os
import uuid
from enum import Enum
from typing import Annotated, Optional

import questionary
import typer
from prettytable.colortable import ColorTable, Themes
from tqdm import tqdm

from memgpt import utils
from memgpt.agent_store.storage import StorageConnector, TableType
from memgpt.config import MemGPTConfig
from memgpt.constants import LLM_MAX_TOKENS, MEMGPT_DIR
from memgpt.credentials import SUPPORTED_AUTH_TYPES, MemGPTCredentials
from memgpt.data_types import EmbeddingConfig, LLMConfig, Source, User
from memgpt.llm_api.anthropic import (
    anthropic_get_model_list,
    antropic_get_model_context_window,
)
from memgpt.llm_api.azure_openai import azure_openai_get_model_list
from memgpt.llm_api.cohere import (
    COHERE_VALID_MODEL_LIST,
    cohere_get_model_context_window,
    cohere_get_model_list,
)
from memgpt.llm_api.google_ai import (
    google_ai_get_model_context_window,
    google_ai_get_model_list,
)
from memgpt.llm_api.llm_api_tools import LLM_API_PROVIDER_OPTIONS
from memgpt.llm_api.openai import openai_get_model_list
from memgpt.local_llm.constants import (
    DEFAULT_ENDPOINTS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WRAPPER_NAME,
)
from memgpt.local_llm.utils import get_available_wrappers
from memgpt.metadata import MetadataStore
from memgpt.models.pydantic_models import HumanModel, PersonaModel
from memgpt.presets.presets import create_preset_from_file
from memgpt.server.utils import shorten_key_middle

app = typer.Typer()


def get_azure_credentials():
    creds = dict(
        azure_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )
    # embedding endpoint and version default to non-embedding
    creds["azure_embedding_endpoint"] = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT", creds["azure_endpoint"])
    creds["azure_embedding_version"] = os.getenv("AZURE_OPENAI_EMBEDDING_VERSION", creds["azure_version"])
    return creds


def get_openai_credentials() -> Optional[str]:
    openai_key = os.getenv("OPENAI_API_KEY", None)
    return openai_key


def get_google_ai_credentials() -> Optional[str]:
    google_ai_key = os.getenv("GOOGLE_AI_API_KEY", None)
    return google_ai_key


def configure_llm_endpoint(config: MemGPTConfig, credentials: MemGPTCredentials):
    # configure model endpoint
    model_endpoint_type, model_endpoint = None, None

    # get default
    default_model_endpoint_type = config.default_llm_config.model_endpoint_type if config.default_embedding_config else None
    if (
        config.default_llm_config
        and config.default_llm_config.model_endpoint_type is not None
        and config.default_llm_config.model_endpoint_type not in [provider for provider in LLM_API_PROVIDER_OPTIONS if provider != "local"]
    ):  # local model
        default_model_endpoint_type = "local"

    provider = questionary.select(
        "Select LLM inference provider:",
        choices=LLM_API_PROVIDER_OPTIONS,
        default=default_model_endpoint_type,
    ).ask()
    if provider is None:
        raise KeyboardInterrupt

    # set: model_endpoint_type, model_endpoint
    if provider == "openai":
        # check for key
        if credentials.openai_key is None:
            # allow key to get pulled from env vars
            openai_api_key = os.getenv("OPENAI_API_KEY", None)
            # if we still can't find it, ask for it as input
            if openai_api_key is None:
                while openai_api_key is None or len(openai_api_key) == 0:
                    # Ask for API key as input
                    openai_api_key = questionary.password(
                        "Enter your OpenAI API key (starts with 'sk-', see https://platform.openai.com/api-keys):"
                    ).ask()
                    if openai_api_key is None:
                        raise KeyboardInterrupt
            credentials.openai_key = openai_api_key
            credentials.save()
        else:
            # Give the user an opportunity to overwrite the key
            openai_api_key = None
            default_input = (
                shorten_key_middle(credentials.openai_key) if credentials.openai_key.startswith("sk-") else credentials.openai_key
            )
            openai_api_key = questionary.password(
                "Enter your OpenAI API key (starts with 'sk-', see https://platform.openai.com/api-keys):",
                default=default_input,
            ).ask()
            if openai_api_key is None:
                raise KeyboardInterrupt
            # If the user modified it, use the new one
            if openai_api_key != default_input:
                credentials.openai_key = openai_api_key
                credentials.save()

        model_endpoint_type = "openai"
        model_endpoint = "https://api.openai.com/v1"
        model_endpoint = questionary.text("Override default endpoint:", default=model_endpoint).ask()
        if model_endpoint is None:
            raise KeyboardInterrupt
        provider = "openai"

    elif provider == "azure":
        # check for necessary vars
        azure_creds = get_azure_credentials()
        if not all([azure_creds["azure_key"], azure_creds["azure_endpoint"], azure_creds["azure_version"]]):
            raise ValueError(
                "Missing environment variables for Azure (see https://memgpt.readme.io/docs/endpoints#azure-openai). Please set then run `memgpt configure` again."
            )
        else:
            credentials.azure_key = azure_creds["azure_key"]
            credentials.azure_version = azure_creds["azure_version"]
            credentials.azure_endpoint = azure_creds["azure_endpoint"]
            if "azure_deployment" in azure_creds:
                credentials.azure_deployment = azure_creds["azure_deployment"]
            credentials.azure_embedding_version = azure_creds["azure_embedding_version"]
            credentials.azure_embedding_endpoint = azure_creds["azure_embedding_endpoint"]
            if "azure_embedding_deployment" in azure_creds:
                credentials.azure_embedding_deployment = azure_creds["azure_embedding_deployment"]
            credentials.save()

        model_endpoint_type = "azure"
        model_endpoint = azure_creds["azure_endpoint"]

    elif provider == "google_ai":

        # check for key
        if credentials.google_ai_key is None:
            # allow key to get pulled from env vars
            google_ai_key = get_google_ai_credentials()
            # if we still can't find it, ask for it as input
            if google_ai_key is None:
                while google_ai_key is None or len(google_ai_key) == 0:
                    # Ask for API key as input
                    google_ai_key = questionary.password(
                        "Enter your Google AI (Gemini) API key (see https://aistudio.google.com/app/apikey):"
                    ).ask()
                    if google_ai_key is None:
                        raise KeyboardInterrupt
            credentials.google_ai_key = google_ai_key
        else:
            # Give the user an opportunity to overwrite the key
            google_ai_key = None
            default_input = shorten_key_middle(credentials.google_ai_key)

            google_ai_key = questionary.password(
                "Enter your Google AI (Gemini) API key (see https://aistudio.google.com/app/apikey):",
                default=default_input,
            ).ask()
            if google_ai_key is None:
                raise KeyboardInterrupt
            # If the user modified it, use the new one
            if google_ai_key != default_input:
                credentials.google_ai_key = google_ai_key

        default_input = os.getenv("GOOGLE_AI_SERVICE_ENDPOINT", None)
        if default_input is None:
            default_input = "generativelanguage"
        google_ai_service_endpoint = questionary.text(
            "Enter your Google AI (Gemini) service endpoint (see https://ai.google.dev/api/rest):",
            default=default_input,
        ).ask()
        credentials.google_ai_service_endpoint = google_ai_service_endpoint

        # write out the credentials
        credentials.save()

        model_endpoint_type = "google_ai"

    elif provider == "anthropic":
        # check for key
        if credentials.anthropic_key is None:
            # allow key to get pulled from env vars
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
            # if we still can't find it, ask for it as input
            if anthropic_api_key is None:
                while anthropic_api_key is None or len(anthropic_api_key) == 0:
                    # Ask for API key as input
                    anthropic_api_key = questionary.password(
                        "Enter your Anthropic API key (starts with 'sk-', see https://console.anthropic.com/settings/keys):"
                    ).ask()
                    if anthropic_api_key is None:
                        raise KeyboardInterrupt
            credentials.anthropic_key = anthropic_api_key
            credentials.save()
        else:
            # Give the user an opportunity to overwrite the key
            anthropic_api_key = None
            default_input = (
                shorten_key_middle(credentials.anthropic_key) if credentials.anthropic_key.startswith("sk-") else credentials.anthropic_key
            )
            anthropic_api_key = questionary.password(
                "Enter your Anthropic API key (starts with 'sk-', see https://console.anthropic.com/settings/keys):",
                default=default_input,
            ).ask()
            if anthropic_api_key is None:
                raise KeyboardInterrupt
            # If the user modified it, use the new one
            if anthropic_api_key != default_input:
                credentials.anthropic_key = anthropic_api_key
                credentials.save()

        model_endpoint_type = "anthropic"
        model_endpoint = "https://api.anthropic.com/v1"
        model_endpoint = questionary.text("Override default endpoint:", default=model_endpoint).ask()
        if model_endpoint is None:
            raise KeyboardInterrupt
        provider = "anthropic"

    elif provider == "cohere":
        # check for key
        if credentials.cohere_key is None:
            # allow key to get pulled from env vars
            cohere_api_key = os.getenv("COHERE_API_KEY", None)
            # if we still can't find it, ask for it as input
            if cohere_api_key is None:
                while cohere_api_key is None or len(cohere_api_key) == 0:
                    # Ask for API key as input
                    cohere_api_key = questionary.password("Enter your Cohere API key (see https://dashboard.cohere.com/api-keys):").ask()
                    if cohere_api_key is None:
                        raise KeyboardInterrupt
            credentials.cohere_key = cohere_api_key
            credentials.save()
        else:
            # Give the user an opportunity to overwrite the key
            cohere_api_key = None
            default_input = (
                shorten_key_middle(credentials.cohere_key) if credentials.cohere_key.startswith("sk-") else credentials.cohere_key
            )
            cohere_api_key = questionary.password(
                "Enter your Cohere API key (see https://dashboard.cohere.com/api-keys):",
                default=default_input,
            ).ask()
            if cohere_api_key is None:
                raise KeyboardInterrupt
            # If the user modified it, use the new one
            if cohere_api_key != default_input:
                credentials.cohere_key = cohere_api_key
                credentials.save()

        model_endpoint_type = "cohere"
        model_endpoint = "https://api.cohere.ai/v1"
        model_endpoint = questionary.text("Override default endpoint:", default=model_endpoint).ask()
        if model_endpoint is None:
            raise KeyboardInterrupt
        provider = "cohere"

    else:  # local models
        # backend_options_old = ["webui", "webui-legacy", "llamacpp", "koboldcpp", "ollama", "lmstudio", "lmstudio-legacy", "vllm", "openai"]
        backend_options = builtins.list(DEFAULT_ENDPOINTS.keys())
        # assert backend_options_old == backend_options, (backend_options_old, backend_options)
        default_model_endpoint_type = None
        if config.default_llm_config and config.default_llm_config.model_endpoint_type in backend_options:
            # set from previous config
            default_model_endpoint_type = config.default_llm_config.model_endpoint_type
        model_endpoint_type = questionary.select(
            "Select LLM backend (select 'openai' if you have an OpenAI compatible proxy):",
            backend_options,
            default=default_model_endpoint_type,
        ).ask()
        if model_endpoint_type is None:
            raise KeyboardInterrupt

        # set default endpoint
        # if OPENAI_API_BASE is set, assume that this is the IP+port the user wanted to use
        default_model_endpoint = os.getenv("OPENAI_API_BASE")
        # if OPENAI_API_BASE is not set, try to pull a default IP+port format from a hardcoded set
        if default_model_endpoint is None:
            if model_endpoint_type in DEFAULT_ENDPOINTS:
                default_model_endpoint = DEFAULT_ENDPOINTS[model_endpoint_type]
                model_endpoint = questionary.text("Enter default endpoint:", default=default_model_endpoint).ask()
                if model_endpoint is None:
                    raise KeyboardInterrupt
                while not utils.is_valid_url(model_endpoint):
                    typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
                    model_endpoint = questionary.text("Enter default endpoint:", default=default_model_endpoint).ask()
                    if model_endpoint is None:
                        raise KeyboardInterrupt
            elif config.default_llm_config and config.default_llm_config.model_endpoint:
                model_endpoint = questionary.text("Enter default endpoint:", default=config.default_llm_config.model_endpoint).ask()
                if model_endpoint is None:
                    raise KeyboardInterrupt
                while not utils.is_valid_url(model_endpoint):
                    typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
                    model_endpoint = questionary.text("Enter default endpoint:", default=config.default_llm_config.model_endpoint).ask()
                    if model_endpoint is None:
                        raise KeyboardInterrupt
            else:
                # default_model_endpoint = None
                model_endpoint = None
                model_endpoint = questionary.text("Enter default endpoint:").ask()
                if model_endpoint is None:
                    raise KeyboardInterrupt
                while not utils.is_valid_url(model_endpoint):
                    typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
                    model_endpoint = questionary.text("Enter default endpoint:").ask()
                    if model_endpoint is None:
                        raise KeyboardInterrupt
        else:
            model_endpoint = default_model_endpoint
        assert model_endpoint, f"Environment variable OPENAI_API_BASE must be set."

    return model_endpoint_type, model_endpoint


def get_model_options(
    credentials: MemGPTCredentials,
    model_endpoint_type: str,
    model_endpoint: str,
    filter_list: bool = True,
    filter_prefix: str = "gpt-",
) -> list:
    try:
        if model_endpoint_type == "openai":
            if credentials.openai_key is None:
                raise ValueError("Missing OpenAI API key")
            fetched_model_options_response = openai_get_model_list(url=model_endpoint, api_key=credentials.openai_key)

            # Filter the list for "gpt" models only
            if filter_list:
                model_options = [obj["id"] for obj in fetched_model_options_response["data"] if obj["id"].startswith(filter_prefix)]
            else:
                model_options = [obj["id"] for obj in fetched_model_options_response["data"]]

        elif model_endpoint_type == "azure":
            if credentials.azure_key is None:
                raise ValueError("Missing Azure key")
            if credentials.azure_version is None:
                raise ValueError("Missing Azure version")
            fetched_model_options_response = azure_openai_get_model_list(
                url=model_endpoint, api_key=credentials.azure_key, api_version=credentials.azure_version
            )

            # Filter the list for "gpt" models only
            if filter_list:
                model_options = [obj["id"] for obj in fetched_model_options_response["data"] if obj["id"].startswith(filter_prefix)]
            else:
                model_options = [obj["id"] for obj in fetched_model_options_response["data"]]

        elif model_endpoint_type == "google_ai":
            if credentials.google_ai_key is None:
                raise ValueError("Missing Google AI API key")
            if credentials.google_ai_service_endpoint is None:
                raise ValueError("Missing Google AI service endpoint")
            model_options = google_ai_get_model_list(
                service_endpoint=credentials.google_ai_service_endpoint, api_key=credentials.google_ai_key
            )
            model_options = [str(m["name"]) for m in model_options]
            model_options = [mo[len("models/") :] if mo.startswith("models/") else mo for mo in model_options]

            # TODO remove manual filtering for gemini-pro
            model_options = [mo for mo in model_options if str(mo).startswith("gemini") and "-pro" in str(mo)]
            # model_options = ["gemini-pro"]

        elif model_endpoint_type == "anthropic":
            if credentials.anthropic_key is None:
                raise ValueError("Missing Anthropic API key")
            fetched_model_options = anthropic_get_model_list(url=model_endpoint, api_key=credentials.anthropic_key)
            model_options = [obj["name"] for obj in fetched_model_options]

        elif model_endpoint_type == "cohere":
            if credentials.cohere_key is None:
                raise ValueError("Missing Cohere API key")
            fetched_model_options = cohere_get_model_list(url=model_endpoint, api_key=credentials.cohere_key)
            model_options = [obj for obj in fetched_model_options]

        else:
            # Attempt to do OpenAI endpoint style model fetching
            # TODO support local auth with api-key header
            if credentials.openllm_auth_type == "bearer_token":
                api_key = credentials.openllm_key
            else:
                api_key = None
            fetched_model_options_response = openai_get_model_list(url=model_endpoint, api_key=api_key, fix_url=True)
            model_options = [obj["id"] for obj in fetched_model_options_response["data"]]
            # NOTE no filtering of local model options

        # list
        return model_options

    except:
        raise Exception(f"Failed to get model list from {model_endpoint}")


def configure_model(config: MemGPTConfig, credentials: MemGPTCredentials, model_endpoint_type: str, model_endpoint: str):
    # set: model, model_wrapper
    model, model_wrapper = None, None
    if model_endpoint_type == "openai" or model_endpoint_type == "azure":
        # Get the model list from the openai / azure endpoint
        hardcoded_model_options = ["gpt-4", "gpt-4-32k", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        fetched_model_options = []
        try:
            fetched_model_options = get_model_options(
                credentials=credentials, model_endpoint_type=model_endpoint_type, model_endpoint=model_endpoint
            )
        except Exception as e:
            # NOTE: if this fails, it means the user's key is probably bad
            typer.secho(
                f"Failed to get model list from {model_endpoint} - make sure your API key and endpoints are correct!", fg=typer.colors.RED
            )
            raise e

        # First ask if the user wants to see the full model list (some may be incompatible)
        see_all_option_str = "[see all options]"
        other_option_str = "[enter model name manually]"

        # Check if the model we have set already is even in the list (informs our default)
        valid_model = config.default_llm_config and config.default_llm_config.model in hardcoded_model_options
        model = questionary.select(
            "Select default model (recommended: gpt-4):",
            choices=hardcoded_model_options + [see_all_option_str, other_option_str],
            default=config.default_llm_config.model if valid_model else hardcoded_model_options[0],
        ).ask()
        if model is None:
            raise KeyboardInterrupt

        # If the user asked for the full list, show it
        if model == see_all_option_str:
            typer.secho(f"Warning: not all models shown are guaranteed to work with MemGPT", fg=typer.colors.RED)
            model = questionary.select(
                "Select default model (recommended: gpt-4):",
                choices=fetched_model_options + [other_option_str],
                default=config.default_llm_config.model if (valid_model and config.default_llm_config) else fetched_model_options[0],
            ).ask()
            if model is None:
                raise KeyboardInterrupt

        # Finally if the user asked to manually input, allow it
        if model == other_option_str:
            model = ""
            while len(model) == 0:
                model = questionary.text(
                    "Enter custom model name:",
                ).ask()
                if model is None:
                    raise KeyboardInterrupt

    elif model_endpoint_type == "google_ai":
        try:
            fetched_model_options = get_model_options(
                credentials=credentials, model_endpoint_type=model_endpoint_type, model_endpoint=model_endpoint
            )
        except Exception as e:
            # NOTE: if this fails, it means the user's key is probably bad
            typer.secho(
                f"Failed to get model list from {model_endpoint} - make sure your API key and endpoints are correct!", fg=typer.colors.RED
            )
            raise e

        model = questionary.select(
            "Select default model:",
            choices=fetched_model_options,
            default=fetched_model_options[0],
        ).ask()
        if model is None:
            raise KeyboardInterrupt

    elif model_endpoint_type == "anthropic":
        try:
            fetched_model_options = get_model_options(
                credentials=credentials, model_endpoint_type=model_endpoint_type, model_endpoint=model_endpoint
            )
        except Exception as e:
            # NOTE: if this fails, it means the user's key is probably bad
            typer.secho(
                f"Failed to get model list from {model_endpoint} - make sure your API key and endpoints are correct!", fg=typer.colors.RED
            )
            raise e

        model = questionary.select(
            "Select default model:",
            choices=fetched_model_options,
            default=fetched_model_options[0],
        ).ask()
        if model is None:
            raise KeyboardInterrupt

    elif model_endpoint_type == "cohere":

        fetched_model_options = []
        try:
            fetched_model_options = get_model_options(
                credentials=credentials, model_endpoint_type=model_endpoint_type, model_endpoint=model_endpoint
            )
        except Exception as e:
            # NOTE: if this fails, it means the user's key is probably bad
            typer.secho(
                f"Failed to get model list from {model_endpoint} - make sure your API key and endpoints are correct!", fg=typer.colors.RED
            )
            raise e

        fetched_model_options = [m["name"] for m in fetched_model_options]
        hardcoded_model_options = [m for m in fetched_model_options if m in COHERE_VALID_MODEL_LIST]

        # First ask if the user wants to see the full model list (some may be incompatible)
        see_all_option_str = "[see all options]"
        other_option_str = "[enter model name manually]"

        # Check if the model we have set already is even in the list (informs our default)
        valid_model = config.default_llm_config.model in hardcoded_model_options
        model = questionary.select(
            "Select default model (recommended: command-r-plus):",
            choices=hardcoded_model_options + [see_all_option_str, other_option_str],
            default=config.default_llm_config.model if valid_model else hardcoded_model_options[0],
        ).ask()
        if model is None:
            raise KeyboardInterrupt

        # If the user asked for the full list, show it
        if model == see_all_option_str:
            typer.secho(f"Warning: not all models shown are guaranteed to work with MemGPT", fg=typer.colors.RED)
            model = questionary.select(
                "Select default model (recommended: command-r-plus):",
                choices=fetched_model_options + [other_option_str],
                default=config.default_llm_config.model if valid_model else fetched_model_options[0],
            ).ask()
            if model is None:
                raise KeyboardInterrupt

        # Finally if the user asked to manually input, allow it
        if model == other_option_str:
            model = ""
            while len(model) == 0:
                model = questionary.text(
                    "Enter custom model name:",
                ).ask()
                if model is None:
                    raise KeyboardInterrupt

    else:  # local models

        # ask about local auth
        if model_endpoint_type in ["groq"]:  # TODO all llm engines under 'local' that will require api keys
            use_local_auth = True
            local_auth_type = "bearer_token"
            local_auth_key = questionary.password(
                "Enter your Groq API key:",
            ).ask()
            if local_auth_key is None:
                raise KeyboardInterrupt
            credentials.openllm_auth_type = local_auth_type
            credentials.openllm_key = local_auth_key
            credentials.save()
        else:
            use_local_auth = questionary.confirm(
                "Is your LLM endpoint authenticated? (default no)",
                default=False,
            ).ask()
            if use_local_auth is None:
                raise KeyboardInterrupt
            if use_local_auth:
                local_auth_type = questionary.select(
                    "What HTTP authentication method does your endpoint require?",
                    choices=SUPPORTED_AUTH_TYPES,
                    default=SUPPORTED_AUTH_TYPES[0],
                ).ask()
                if local_auth_type is None:
                    raise KeyboardInterrupt
                local_auth_key = questionary.password(
                    "Enter your authentication key:",
                ).ask()
                if local_auth_key is None:
                    raise KeyboardInterrupt
                # credentials = MemGPTCredentials.load()
                credentials.openllm_auth_type = local_auth_type
                credentials.openllm_key = local_auth_key
                credentials.save()

        # ollama also needs model type
        if model_endpoint_type == "ollama":
            default_model = (
                config.default_llm_config.model
                if config.default_llm_config and config.default_llm_config.model_endpoint_type == "ollama"
                else DEFAULT_OLLAMA_MODEL
            )
            model = questionary.text(
                "Enter default model name (required for Ollama, see: https://memgpt.readme.io/docs/ollama):",
                default=default_model,
            ).ask()
            if model is None:
                raise KeyboardInterrupt
            model = None if len(model) == 0 else model

        default_model = (
            config.default_llm_config.model if config.default_llm_config and config.default_llm_config.model_endpoint_type == "vllm" else ""
        )

        # vllm needs huggingface model tag
        if model_endpoint_type in ["vllm", "groq"]:
            try:
                # Don't filter model list for vLLM since model list is likely much smaller than OpenAI/Azure endpoint
                # + probably has custom model names
                # TODO support local auth
                model_options = get_model_options(
                    credentials=credentials, model_endpoint_type=model_endpoint_type, model_endpoint=model_endpoint
                )
            except:
                print(f"Failed to get model list from {model_endpoint}, using defaults")
                model_options = None

            # If we got model options from vLLM endpoint, allow selection + custom input
            if model_options is not None:
                other_option_str = "other (enter name)"
                valid_model = config.default_llm_config.model in model_options
                model_options.append(other_option_str)
                model = questionary.select(
                    "Select default model:",
                    choices=model_options,
                    default=config.default_llm_config.model if valid_model else model_options[0],
                ).ask()
                if model is None:
                    raise KeyboardInterrupt

                # If we got custom input, ask for raw input
                if model == other_option_str:
                    model = questionary.text(
                        "Enter HuggingFace model tag (e.g. ehartford/dolphin-2.2.1-mistral-7b):",
                        default=default_model,
                    ).ask()
                    if model is None:
                        raise KeyboardInterrupt
                    # TODO allow empty string for input?
                    model = None if len(model) == 0 else model

            else:
                model = questionary.text(
                    "Enter HuggingFace model tag (e.g. ehartford/dolphin-2.2.1-mistral-7b):",
                    default=default_model,
                ).ask()
                if model is None:
                    raise KeyboardInterrupt
                model = None if len(model) == 0 else model

        # model wrapper
        available_model_wrappers = builtins.list(get_available_wrappers().keys())
        model_wrapper = questionary.select(
            f"Select default model wrapper (recommended: {DEFAULT_WRAPPER_NAME}):",
            choices=available_model_wrappers,
            default=DEFAULT_WRAPPER_NAME,
        ).ask()
        if model_wrapper is None:
            raise KeyboardInterrupt

    # set: context_window
    if str(model) not in LLM_MAX_TOKENS:

        context_length_options = [
            str(2**12),  # 4096
            str(2**13),  # 8192
            str(2**14),  # 16384
            str(2**15),  # 32768
            str(2**18),  # 262144
            "custom",  # enter yourself
        ]

        if model_endpoint_type == "google_ai":
            try:
                fetched_context_window = str(
                    google_ai_get_model_context_window(
                        service_endpoint=credentials.google_ai_service_endpoint, api_key=credentials.google_ai_key, model=model
                    )
                )
                print(f"Got context window {fetched_context_window} for model {model} (from Google API)")
                context_length_options = [
                    fetched_context_window,
                    "custom",
                ]
            except Exception as e:
                print(f"Failed to get model details for model '{model}' on Google AI API ({str(e)})")

            context_window_input = questionary.select(
                "Select your model's context window (see https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning#gemini-model-versions):",
                choices=context_length_options,
                default=context_length_options[0],
            ).ask()
            if context_window_input is None:
                raise KeyboardInterrupt

        elif model_endpoint_type == "anthropic":
            try:
                fetched_context_window = str(
                    antropic_get_model_context_window(url=model_endpoint, api_key=credentials.anthropic_key, model=model)
                )
                print(f"Got context window {fetched_context_window} for model {model}")
                context_length_options = [
                    fetched_context_window,
                    "custom",
                ]
            except Exception as e:
                print(f"Failed to get model details for model '{model}' ({str(e)})")

            context_window_input = questionary.select(
                "Select your model's context window (see https://docs.anthropic.com/claude/docs/models-overview):",
                choices=context_length_options,
                default=context_length_options[0],
            ).ask()
            if context_window_input is None:
                raise KeyboardInterrupt

        elif model_endpoint_type == "cohere":
            try:
                fetched_context_window = str(
                    cohere_get_model_context_window(url=model_endpoint, api_key=credentials.cohere_key, model=model)
                )
                print(f"Got context window {fetched_context_window} for model {model}")
                context_length_options = [
                    fetched_context_window,
                    "custom",
                ]
            except Exception as e:
                print(f"Failed to get model details for model '{model}' ({str(e)})")

            context_window_input = questionary.select(
                "Select your model's context window (see https://docs.cohere.com/docs/command-r):",
                choices=context_length_options,
                default=context_length_options[0],
            ).ask()
            if context_window_input is None:
                raise KeyboardInterrupt

        else:

            # Ask the user to specify the context length
            context_window_input = questionary.select(
                "Select your model's context window (for Mistral 7B models, this is probably 8k / 8192):",
                choices=context_length_options,
                default=str(LLM_MAX_TOKENS["DEFAULT"]),
            ).ask()
            if context_window_input is None:
                raise KeyboardInterrupt

        # If custom, ask for input
        if context_window_input == "custom":
            while True:
                context_window_input = questionary.text("Enter context window (e.g. 8192)").ask()
                if context_window_input is None:
                    raise KeyboardInterrupt
                try:
                    context_window = int(context_window_input)
                    break
                except ValueError:
                    print(f"Context window must be a valid integer")
        else:
            context_window = int(context_window_input)
    else:
        # Pull the context length from the models
        context_window = int(LLM_MAX_TOKENS[str(model)])
    return model, model_wrapper, context_window


def configure_embedding_endpoint(config: MemGPTConfig, credentials: MemGPTCredentials):
    # configure embedding endpoint

    default_embedding_endpoint_type = config.default_embedding_config.embedding_endpoint_type if config.default_embedding_config else None

    embedding_endpoint_type, embedding_endpoint, embedding_dim, embedding_model = None, None, None, None
    embedding_provider = questionary.select(
        "Select embedding provider:", choices=["openai", "azure", "hugging-face", "local"], default=default_embedding_endpoint_type
    ).ask()
    if embedding_provider is None:
        raise KeyboardInterrupt

    if embedding_provider == "openai":
        # check for key
        if credentials.openai_key is None:
            # allow key to get pulled from env vars
            openai_api_key = os.getenv("OPENAI_API_KEY", None)
            if openai_api_key is None:
                # if we still can't find it, ask for it as input
                while openai_api_key is None or len(openai_api_key) == 0:
                    # Ask for API key as input
                    openai_api_key = questionary.password(
                        "Enter your OpenAI API key (starts with 'sk-', see https://platform.openai.com/api-keys):"
                    ).ask()
                    if openai_api_key is None:
                        raise KeyboardInterrupt
                credentials.openai_key = openai_api_key
                credentials.save()

        embedding_endpoint_type = "openai"
        embedding_endpoint = "https://api.openai.com/v1"
        embedding_dim = 1536
        embedding_model = "text-embedding-ada-002"

    elif embedding_provider == "azure":
        # check for necessary vars
        azure_creds = get_azure_credentials()
        if not all([azure_creds["azure_key"], azure_creds["azure_embedding_endpoint"], azure_creds["azure_embedding_version"]]):
            raise ValueError(
                "Missing environment variables for Azure (see https://memgpt.readme.io/docs/endpoints#azure-openai). Please set then run `memgpt configure` again."
            )
        credentials.azure_key = azure_creds["azure_key"]
        credentials.azure_version = azure_creds["azure_version"]
        credentials.azure_embedding_endpoint = azure_creds["azure_embedding_endpoint"]
        credentials.save()

        embedding_endpoint_type = "azure"
        embedding_endpoint = azure_creds["azure_embedding_endpoint"]
        embedding_dim = 1536
        embedding_model = "text-embedding-ada-002"

    elif embedding_provider == "hugging-face":
        # configure hugging face embedding endpoint (https://github.com/huggingface/text-embeddings-inference)
        # supports custom model/endpoints
        embedding_endpoint_type = "hugging-face"
        embedding_endpoint = None

        # get endpoint
        embedding_endpoint = questionary.text("Enter default endpoint:").ask()
        if embedding_endpoint is None:
            raise KeyboardInterrupt
        while not utils.is_valid_url(embedding_endpoint):
            typer.secho(f"Endpoint must be a valid address", fg=typer.colors.YELLOW)
            embedding_endpoint = questionary.text("Enter default endpoint:").ask()
            if embedding_endpoint is None:
                raise KeyboardInterrupt

        # get model type
        default_embedding_model = (
            config.default_embedding_config.embedding_model if config.default_embedding_config else "BAAI/bge-large-en-v1.5"
        )
        embedding_model = questionary.text(
            "Enter HuggingFace model tag (e.g. BAAI/bge-large-en-v1.5):",
            default=default_embedding_model,
        ).ask()
        if embedding_model is None:
            raise KeyboardInterrupt

        # get model dimentions
        default_embedding_dim = config.default_embedding_config.embedding_dim if config.default_embedding_config else "1024"
        embedding_dim = questionary.text("Enter embedding model dimentions (e.g. 1024):", default=str(default_embedding_dim)).ask()
        if embedding_dim is None:
            raise KeyboardInterrupt
        try:
            embedding_dim = int(embedding_dim)
        except Exception:
            raise ValueError(f"Failed to cast {embedding_dim} to integer.")
    else:  # local models
        embedding_endpoint_type = "local"
        embedding_endpoint = None
        embedding_model = "BAAI/bge-small-en-v1.5"
        embedding_dim = 384

    return embedding_endpoint_type, embedding_endpoint, embedding_dim, embedding_model


def configure_archival_storage(config: MemGPTConfig, credentials: MemGPTCredentials):
    # Configure archival storage backend
    archival_storage_options = ["postgres", "chroma"]
    archival_storage_type = questionary.select(
        "Select storage backend for archival data:", archival_storage_options, default=config.archival_storage_type
    ).ask()
    if archival_storage_type is None:
        raise KeyboardInterrupt
    archival_storage_uri, archival_storage_path = config.archival_storage_uri, config.archival_storage_path

    # configure postgres
    if archival_storage_type == "postgres":
        archival_storage_uri = questionary.text(
            "Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}):",
            default=config.archival_storage_uri if config.archival_storage_uri else "",
        ).ask()
        if archival_storage_uri is None:
            raise KeyboardInterrupt

    # TODO: add back
    ## configure lancedb
    # if archival_storage_type == "lancedb":
    #    archival_storage_uri = questionary.text(
    #        "Enter lanncedb connection string (e.g. ./.lancedb",
    #        default=config.archival_storage_uri if config.archival_storage_uri else "./.lancedb",
    #    ).ask()

    # configure chroma
    if archival_storage_type == "chroma":
        chroma_type = questionary.select("Select chroma backend:", ["http", "persistent"], default="persistent").ask()
        if chroma_type is None:
            raise KeyboardInterrupt
        if chroma_type == "http":
            archival_storage_uri = questionary.text("Enter chroma ip (e.g. localhost:8000):", default="localhost:8000").ask()
            if archival_storage_uri is None:
                raise KeyboardInterrupt
        if chroma_type == "persistent":
            archival_storage_path = os.path.join(MEMGPT_DIR, "chroma")

    return archival_storage_type, archival_storage_uri, archival_storage_path

    # TODO: allow configuring embedding model


def configure_recall_storage(config: MemGPTConfig, credentials: MemGPTCredentials):
    # Configure recall storage backend
    recall_storage_options = ["sqlite", "postgres"]
    recall_storage_type = questionary.select(
        "Select storage backend for recall data:", recall_storage_options, default=config.recall_storage_type
    ).ask()
    if recall_storage_type is None:
        raise KeyboardInterrupt
    recall_storage_uri, recall_storage_path = config.recall_storage_uri, config.recall_storage_path
    # configure postgres
    if recall_storage_type == "postgres":
        recall_storage_uri = questionary.text(
            "Enter postgres connection string (e.g. postgresql+pg8000://{user}:{password}@{ip}:5432/{database}):",
            default=config.recall_storage_uri if config.recall_storage_uri else "",
        ).ask()
        if recall_storage_uri is None:
            raise KeyboardInterrupt

    return recall_storage_type, recall_storage_uri, recall_storage_path


@app.command()
def configure():
    """Updates default MemGPT configurations

    This function and quickstart should be the ONLY place where MemGPTConfig.save() is called
    """

    # check credentials
    credentials = MemGPTCredentials.load()
    openai_key = get_openai_credentials()

    MemGPTConfig.create_config_dir()

    # Will pre-populate with defaults, or what the user previously set
    config = MemGPTConfig.load()
    try:
        model_endpoint_type, model_endpoint = configure_llm_endpoint(
            config=config,
            credentials=credentials,
        )
        model, model_wrapper, context_window = configure_model(
            config=config,
            credentials=credentials,
            model_endpoint_type=str(model_endpoint_type),
            model_endpoint=str(model_endpoint),
        )
        embedding_endpoint_type, embedding_endpoint, embedding_dim, embedding_model = configure_embedding_endpoint(
            config=config,
            credentials=credentials,
        )
        archival_storage_type, archival_storage_uri, archival_storage_path = configure_archival_storage(
            config=config,
            credentials=credentials,
        )
        recall_storage_type, recall_storage_uri, recall_storage_path = configure_recall_storage(
            config=config,
            credentials=credentials,
        )
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        return

    # openai key might have gotten added along the way
    openai_key = credentials.openai_key if credentials.openai_key is not None else openai_key

    # TODO: remove most of this (deplicated with User table)
    config = MemGPTConfig(
        default_llm_config=LLMConfig(
            model=model,
            model_endpoint=model_endpoint,
            model_endpoint_type=model_endpoint_type,
            model_wrapper=model_wrapper,
            context_window=context_window,
        ),
        default_embedding_config=EmbeddingConfig(
            embedding_endpoint_type=embedding_endpoint_type,
            embedding_endpoint=embedding_endpoint,
            embedding_dim=embedding_dim,
            embedding_model=embedding_model,
        ),
        # storage
        archival_storage_type=archival_storage_type,
        archival_storage_uri=archival_storage_uri,
        archival_storage_path=archival_storage_path,
        # recall storage
        recall_storage_type=recall_storage_type,
        recall_storage_uri=recall_storage_uri,
        recall_storage_path=recall_storage_path,
        # metadata storage (currently forced to match recall storage)
        metadata_storage_type=recall_storage_type,
        metadata_storage_uri=recall_storage_uri,
        metadata_storage_path=recall_storage_path,
    )

    typer.secho(f"📖 Saving config to {config.config_path}", fg=typer.colors.GREEN)
    config.save()

    # create user records
    ms = MetadataStore(config)
    user_id = uuid.UUID(config.anon_clientid)
    user = User(
        id=uuid.UUID(config.anon_clientid),
    )
    if ms.get_user(user_id):
        # update user
        ms.update_user(user)
    else:
        ms.create_user(user)

    # create preset records in metadata store
    from memgpt.presets.presets import add_default_presets

    add_default_presets(user_id, ms)


class ListChoice(str, Enum):
    agents = "agents"
    humans = "humans"
    personas = "personas"
    sources = "sources"
    presets = "presets"


@app.command()
def list(arg: Annotated[ListChoice, typer.Argument]):
    config = MemGPTConfig.load()
    ms = MetadataStore(config)
    user_id = uuid.UUID(config.anon_clientid)
    table = ColorTable(theme=Themes.OCEAN)
    if arg == ListChoice.agents:
        """List all agents"""
        table.field_names = ["Name", "LLM Model", "Embedding Model", "Embedding Dim", "Persona", "Human", "Data Source", "Create Time"]
        for agent in tqdm(ms.list_agents(user_id=user_id)):
            source_ids = ms.list_attached_sources(agent_id=agent.id)
            assert all([source_id is not None and isinstance(source_id, uuid.UUID) for source_id in source_ids])
            sources = [ms.get_source(source_id=source_id) for source_id in source_ids]
            assert all([source is not None and isinstance(source, Source)] for source in sources)
            source_names = [source.name for source in sources if source is not None]
            table.add_row(
                [
                    agent.name,
                    agent.llm_config.model,
                    agent.embedding_config.embedding_model,
                    agent.embedding_config.embedding_dim,
                    agent.persona,
                    agent.human,
                    ",".join(source_names),
                    utils.format_datetime(agent.created_at),
                ]
            )
        print(table)
    elif arg == ListChoice.humans:
        """List all humans"""
        table.field_names = ["Name", "Text"]
        for human in ms.list_humans(user_id=user_id):
            table.add_row([human.name, human.text.replace("\n", "")[:100]])
        print(table)
    elif arg == ListChoice.personas:
        """List all personas"""
        table.field_names = ["Name", "Text"]
        for persona in ms.list_personas(user_id=user_id):
            table.add_row([persona.name, persona.text.replace("\n", "")[:100]])
        print(table)
    elif arg == ListChoice.sources:
        """List all data sources"""

        # create table
        table.field_names = ["Name", "Description", "Embedding Model", "Embedding Dim", "Created At", "Agents"]
        # TODO: eventually look accross all storage connections
        # TODO: add data source stats
        # TODO: connect to agents

        # get all sources
        for source in ms.list_sources(user_id=user_id):
            # get attached agents
            agent_ids = ms.list_attached_agents(source_id=source.id)
            agent_states = [ms.get_agent(agent_id=agent_id) for agent_id in agent_ids]
            agent_names = [agent_state.name for agent_state in agent_states if agent_state is not None]

            table.add_row(
                [
                    source.name,
                    source.description,
                    source.embedding_model,
                    source.embedding_dim,
                    utils.format_datetime(source.created_at),
                    ",".join(agent_names),
                ]
            )

        print(table)
    elif arg == ListChoice.presets:
        """List all available presets"""
        table.field_names = ["Name", "Description", "Sources", "Functions"]
        for preset in ms.list_presets(user_id=user_id):
            sources = ms.get_preset_sources(preset_id=preset.id)
            table.add_row(
                [
                    preset.name,
                    preset.description,
                    ",".join([source.name for source in sources]),
                    # json.dumps(preset.functions_schema, indent=4)
                    ",\n".join([f["name"] for f in preset.functions_schema]),
                ]
            )
        print(table)
    else:
        raise ValueError(f"Unknown argument {arg}")


@app.command()
def add(
    option: str,  # [human, persona]
    name: Annotated[str, typer.Option(help="Name of human/persona")],
    text: Annotated[Optional[str], typer.Option(help="Text of human/persona")] = None,
    filename: Annotated[Optional[str], typer.Option("-f", help="Specify filename")] = None,
):
    """Add a person/human"""
    config = MemGPTConfig.load()
    user_id = uuid.UUID(config.anon_clientid)
    ms = MetadataStore(config)
    if filename:  # read from file
        assert text is None, "Cannot specify both text and filename"
        with open(filename, "r") as f:
            text = f.read()
    if option == "persona":
        persona = ms.get_persona(name=name, user_id=user_id)
        if persona:
            # config if user wants to overwrite
            if not questionary.confirm(f"Persona {name} already exists. Overwrite?").ask():
                return
            persona.text = text
            ms.update_persona(persona)
        else:
            persona = PersonaModel(name=name, text=text, user_id=user_id)
            ms.add_persona(persona)

    elif option == "human":
        human = ms.get_human(name=name, user_id=user_id)
        if human:
            # config if user wants to overwrite
            if not questionary.confirm(f"Human {name} already exists. Overwrite?").ask():
                return
            human.text = text
            ms.update_human(human)
        else:
            human = HumanModel(name=name, text=text, user_id=user_id)
            ms.add_human(HumanModel(name=name, text=text, user_id=user_id))
    elif option == "preset":
        assert filename, "Must specify filename for preset"
        create_preset_from_file(filename, name, user_id, ms)
    else:
        raise ValueError(f"Unknown kind {option}")


@app.command()
def delete(option: str, name: str):
    """Delete a source from the archival memory."""

    config = MemGPTConfig.load()
    user_id = uuid.UUID(config.anon_clientid)
    ms = MetadataStore(config)
    assert ms.get_user(user_id=user_id), f"User {user_id} does not exist"

    try:
        # delete from metadata
        if option == "source":
            # delete metadata
            source = ms.get_source(source_name=name, user_id=user_id)
            assert source is not None, f"Source {name} does not exist"
            ms.delete_source(source_id=source.id)

            # delete from passages
            conn = StorageConnector.get_storage_connector(TableType.PASSAGES, config, user_id=user_id)
            conn.delete({"data_source": name})

            assert (
                conn.get_all({"data_source": name}) == []
            ), f"Expected no passages with source {name}, but got {conn.get_all({'data_source': name})}"

            # TODO: should we also delete from agents?
        elif option == "agent":
            agent = ms.get_agent(agent_name=name, user_id=user_id)
            assert agent is not None, f"Agent {name} for user_id {user_id} does not exist"

            # recall memory
            recall_conn = StorageConnector.get_storage_connector(TableType.RECALL_MEMORY, config, user_id=user_id, agent_id=agent.id)
            recall_conn.delete({"agent_id": agent.id})

            # archival memory
            archival_conn = StorageConnector.get_storage_connector(TableType.ARCHIVAL_MEMORY, config, user_id=user_id, agent_id=agent.id)
            archival_conn.delete({"agent_id": agent.id})

            # metadata
            ms.delete_agent(agent_id=agent.id)

        elif option == "human":
            human = ms.get_human(name=name, user_id=user_id)
            assert human is not None, f"Human {name} does not exist"
            ms.delete_human(name=name, user_id=user_id)
        elif option == "persona":
            persona = ms.get_persona(name=name, user_id=user_id)
            assert persona is not None, f"Persona {name} does not exist"
            ms.delete_persona(name=name, user_id=user_id)
            assert ms.get_persona(name=name, user_id=user_id) is None, f"Persona {name} still exists"
        elif option == "preset":
            preset = ms.get_preset(name=name, user_id=user_id)
            assert preset is not None, f"Preset {name} does not exist"
            ms.delete_preset(name=name, user_id=user_id)
        else:
            raise ValueError(f"Option {option} not implemented")

        typer.secho(f"Deleted {option} '{name}'", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"Failed to delete {option}'{name}'\n{e}", fg=typer.colors.RED)
