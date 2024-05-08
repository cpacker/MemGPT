from pytest import fixture
import uuid
from httpx import Client

from memgpt.server import app
from memgpt.metadata import MetadataStore, User, SecureTokenModel
from memgpt.server.rest_api.auth.security import Security, EncryptedSecureKey, RawSecureKey

class TestAuthUnit:
    """localized unit tests for the security enhancements"""

    @fixture
    def user_storage(self, tmpdir) -> "MetadataStore":

        class ConfigStub:
            metadata_storage_type = "sqlite"
            metadata_storage_path = str(tmpdir)
        metadata_store = MetadataStore(ConfigStub())
        user_id = uuid.uuid4()
        metadata_store.create_user(User(id=user_id))
        return user_id, metadata_store

    def _generate_new_secure_key(self, user_id:"uuid.UUID", db:"MetadataStore"):
        """Isolating this since seeding it would be a bear right now."""
        with db.session_maker() as session:
            api_key, secure_token = SecureTokenModel.create(user_id=user_id, session=session)
        return api_key, secure_token

    @fixture
    def key_pair(self, user_storage):
        user_id, db = user_storage
        return self._generate_new_secure_key(user_id, db)

    def test_generate_new_secure_key(self, key_pair):
        api_key, secure_token = key_pair
        assert api_key
        assert api_key.startswith("sks-")
        assert not secure_token.token == api_key

    def test_valid_api_key_validates(self, key_pair):
        api_key, secure_token = key_pair
        security = Security()
        raw_key = security.decode_raw_api_key(api_key)
        control_key = EncryptedSecureKey(key_id=secure_token.id, encrypted_secret=secure_token.token)
        assert security.verify_secure_key(raw_key, control_key)

    def test_invalid_api_key_fails_validation(self, key_pair):
        api_key, secure_token = key_pair
        security = Security()
        bad_key = security.encode_raw_secure_key(RawSecureKey(key_id=secure_token.id, raw_secret="this is a bad key"))
        raw_key = security.decode_raw_api_key(bad_key)
        control_key = EncryptedSecureKey(key_id=secure_token.id, encrypted_secret=secure_token.token)
        # bad token
        assert not security.verify_secure_key(raw_key, control_key)
        breakpoint()
        # bad id (sanity check)
        new_bad_key = security.decode_raw_api_key(api_key=api_key)
        new_bad_key.key_id = secure_token.id + 1
        assert not security.verify_secure_key(new_bad_key, control_key)



class TestAuth:

    def test_login(self):

        client = Client(app=app, base_url="/v1")

        response = client.post("/login", json={"username": "test", "password": "test"})
        assert response.status_code == 200