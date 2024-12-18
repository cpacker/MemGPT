import os
import shutil
import pytest
import uuid

from letta.credentials import LettaCredentials

# this must not bee imported from credentials.py, because it will ensure that no one will be removed.
# TODO: if 31facc13565df7f10eeb14adb26041e8fc806132 commit will be accepted, replace this dict comprehantion by import:
# from letta.credentials import PROVIDERS_FEALDS as original_provider_fealds

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


@pytest.fixture
def set_test_credentials_path():
  # save previous credentials_path, and set test dir
  credentials_path = os.getenv("MEMGPT_CREDENTIALS_PATH")

  # prepare and set test_credentials_path
  test_credentials_path = '/tmp/test_letta_credentials'
  if not os.path.exists(test_credentials_path):
    os.makedirs(test_credentials_path)
  os.environ["MEMGPT_CREDENTIALS_PATH"] = test_credentials_path
  
  try:
    # Yield control to the test
    yield
  finally:
    # Ensure this runs no matter what
    if credentials_path is None:
      delete(os.environ["MEMGPT_CREDENTIALS_PATH"])
    else:
      os.environ["MEMGPT_CREDENTIALS_PATH"] = credentials_path
      
    if os.path.exists(test_credentials_path):
      shutil.rmtree(test_credentials_path)


def test_save_load_cicle(set_test_credentials_path):
  # prepare test data
  cred_kwargs = {"credei"}
  for provider, feald_name in PROVIDERS_FEALDS.items():
    cred_kwargs[f"{provider}_{feald_name}"] = uuid.uuid4()

  # create credentials object and save it
  credentials = LettaCredentials(**cred_kwargs)
  credentials.save()

  # check that loading data runs correctly
  loaded_credentials = LettaCredentials.load()
  for cred_prop_name, value in cred_kwargs.items():
    assert getattribute(loaded_credentials, cred_prop_name) == value
