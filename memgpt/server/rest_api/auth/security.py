from typing import Tuple
from base64 import b64encode, b64decode
import uuid
from pydantic import BaseModel
from passlib.context import CryptContext

from memgpt.settings import settings


class EncryptedSecureKey(BaseModel):
    key_id: int
    encrypted_secret: str

class RawSecureKey(BaseModel):
    key_id: int
    raw_secret: str

class Security:
    secret_context: CryptContext
    algorithm: str = "HS256"

    def __init__(self):
        self.secret_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.settings = settings

    def get_hashed_pair(self) -> Tuple[str, str]:
        """creates a one-way hashed pair: a secret, and the hash of that secret

        Returns:
            Tuple[str, str]: a secret and its hash
        """
        secret = uuid.uuid4().hex
        return secret, self.secret_context.hash(secret)

    def decode_raw_api_key(self, api_key: str) -> "RawSecureKey":
        """parses the encoded api key and returns the key ID and encrypted secret"""
        try:
            encoded = api_key.split("sks-")[1]
        except IndexError:
            raise ValueError("Not a secure api key")
        # add back in the vanity-removed padding
        id, secret = b64decode(encoded.encode("utf-8") + b"==").decode().split(":")
        return RawSecureKey(key_id=int(id), raw_secret=secret)

    def encode_raw_secure_key(self, secure_key: "RawSecureKey") -> str:
        """encodes the key ID and encrypted secret into an API key designated secure"""
        plaintext = f"{secure_key.key_id}:{secure_key.raw_secret}"
        encoded = b64encode(plaintext.encode()).decode().strip("=")
        return f"sks-{encoded}"

    def verify_secure_key(self, candidate_key: RawSecureKey, target_key:EncryptedSecureKey) -> bool:
        """verifies the candidate secret against the encrypted secret in the secure key
        Returns:
            bool: True if the secret matches the encrypted secret, False otherwise
        """
        return all(
            ((candidate_key.key_id == target_key.key_id),
            self.secret_context.verify(candidate_key.raw_secret, target_key.encrypted_secret),))


    def create_jwt_auth_token(self, data: dict) -> str:
        """creates an authentication token
        Args:
            data (dict): data to be stored in the token
        Returns:
            str: authentication token
        """
        raise NotImplementedError

    def create_jwt_refresh_token(self, data: dict) -> str:
        """creates a refresh token
        Args:
            data (dict): data to be stored in the token

        Returns:
            str: refresh token
        """
        raise NotImplementedError

    def _create_jwt_access_token(self, data: dict, expires: Tuple[str, int]) -> str:
        """creates a JWT access token
        Args:
            data (dict): data to be stored in the token
            expires (Tuple[str, int]): expiration time for the token in format (time unit, time value) e.g. ("hours", 1)

        Returns:
            str: JWT access token
        """
        raise NotImplementedError

    def read_jwt_access_token(self, token: str) -> dict:
        """reads the data stored in the access token"""
        raise NotImplementedError
