import configparser
import os
from dataclasses import dataclass
from typing import Optional

from letta.config import get_field, set_field
from letta.constants import LETTA_DIR

SUPPORTED_AUTH_TYPES = ["bearer_token", "api_key"]

PROVIDERS_FEALDS = {
            "openai": "auth_type",
            "openai": "key",
    
            "azure": "auth_type",
            "azure": "key",
            "azure": "version", 
            "azure": "endpoint",
            "azure": "deployment", 
            "azure": "embedding_version",
            "azure": "embedding_endpoint",
            "azure": "embedding_deployment",
    
            "google_ai": "key",
    
            "anthropic": "key",
    
            "cohere": "key",
    
            "groq": "key",
    
            "openllm": "auth_type", 
            "openllm": "key",
        }

@dataclass
class LettaCredentials:
    # credentials for Letta
    credentials_path: str = os.path.join(LETTA_DIR, "credentials")

    # openai config
    openai_auth_type: str = "bearer_token"
    openai_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    # gemini config
    google_ai_key: Optional[str] = None
    google_ai_service_endpoint: Optional[str] = None

    # anthropic config
    anthropic_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # cohere config
    cohere_key: Optional[str] = None

    # azure config
    azure_auth_type: str = "api_key"
    azure_key: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY")

    # groq config
    groq_key: Optional[str] = os.getenv("GROQ_API_KEY")

    # base llm / model
    azure_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    # embeddings
    azure_embedding_version: Optional[str] = None
    azure_embedding_endpoint: Optional[str] = None
    azure_embedding_deployment: Optional[str] = None

    # custom llm API config
    openllm_auth_type: Optional[str] = None
    openllm_key: Optional[str] = None

    @classmethod
    def load(cls) -> "LettaCredentials":
        config = configparser.ConfigParser()

        # allow overriding with env variables
        
        credentials_path = os.getenv("MEMGPT_CREDENTIALS_PATH")
        if not credentials_path:
            credentials_path = LettaCredentials.credentials_path

        if not os.path.exists(credentials_path):

            # create new config
            config = cls(credentials_path=credentials_path)
            config.save()  # save updated config
            return config
            
        # read existing credentials
        config.read(credentials_path)

        config_dict = {}
        for provider, field_name in PROVIDERS_FEALDS.items():
            value = get_field(config, provider, field_name)
            if value is not None:
                config_dict[f"{provider}_{field_name}"] = value
                    
        config_dict["credentials_path"] = credentials_path,
        return cls(**config_dict)


    def save(self):
        pass

        config = configparser.ConfigParser()
        # openai config

        for provider, field_name in PROVIDERS_FEALDS.items():
            set_field(
                config, provider, field_name, 
                self.__getattribute__(f"{provider}_{field_name}")
            )

        if not os.path.exists(LETTA_DIR):
            os.makedirs(LETTA_DIR, exist_ok=True)
        with open(self.credentials_path, "w", encoding="utf-8") as f:
            config.write(f)

    @staticmethod
    def exists():
        # allow overriding with env variables
        if os.getenv("MEMGPT_CREDENTIALS_PATH"):
            credentials_path = os.getenv("MEMGPT_CREDENTIALS_PATH")
        else:
            credentials_path = LettaCredentials.credentials_path

        assert not os.path.isdir(credentials_path), f"Credentials path {credentials_path} cannot be set to a directory."
        return os.path.exists(credentials_path)
