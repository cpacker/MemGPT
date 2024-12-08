import os
import shutil
import pytest
import uuid

from letta.credentials import LettaCredentials as Credentials, PROVIDERS_FEALDS


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
