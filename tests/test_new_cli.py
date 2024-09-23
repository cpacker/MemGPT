# TODO: fix later

# import os
# import random
# import string
# import unittest.mock
#
# import pytest
#
# from letta.cli.cli_config import add, delete, list
# from letta.config import LettaConfig
# from letta.credentials import LettaCredentials
# from tests.utils import create_config
#
#
# def _reset_config():
#
#    if os.getenv("OPENAI_API_KEY"):
#        create_config("openai")
#        credentials = LettaCredentials(
#            openai_key=os.getenv("OPENAI_API_KEY"),
#        )
#    else:  # hosted
#        create_config("letta_hosted")
#        credentials = LettaCredentials()
#
#    config = LettaConfig.load()
#    config.save()
#    credentials.save()
#    print("_reset_config :: ", config.config_path)
#
#
# @pytest.mark.skip(reason="This is a helper function.")
# def generate_random_string(length):
#    characters = string.ascii_letters + string.digits
#    random_string = "".join(random.choices(characters, k=length))
#    return random_string
#
#
# @pytest.mark.skip(reason="Ensures LocalClient is used during testing.")
# def unset_env_variables():
#    server_url = os.environ.pop("MEMGPT_BASE_URL", None)
#    token = os.environ.pop("MEMGPT_SERVER_PASS", None)
#    return server_url, token
#
#
# @pytest.mark.skip(reason="Set env variables back to values before test.")
# def reset_env_variables(server_url, token):
#    if server_url is not None:
#        os.environ["MEMGPT_BASE_URL"] = server_url
#    if token is not None:
#        os.environ["MEMGPT_SERVER_PASS"] = token
#
#
# def test_crud_human(capsys):
#    _reset_config()
#
#    server_url, token = unset_env_variables()
#
#    # Initialize values that won't interfere with existing ones
#    human_1 = generate_random_string(16)
#    text_1 = generate_random_string(32)
#    human_2 = generate_random_string(16)
#    text_2 = generate_random_string(32)
#    text_3 = generate_random_string(32)
#
#    # Add inital human
#    add("human", human_1, text_1)
#
#    # Expect inital human to be listed
#    list("humans")
#    captured = capsys.readouterr()
#    output = captured.out[captured.out.find(human_1) :]
#
#    assert human_1 in output
#    assert text_1 in output
#
#    # Add second human
#    add("human", human_2, text_2)
#
#    # Expect to see second human
#    list("humans")
#    captured = capsys.readouterr()
#    output = captured.out[captured.out.find(human_1) :]
#
#    assert human_1 in output
#    assert text_1 in output
#    assert human_2 in output
#    assert text_2 in output
#
#    with unittest.mock.patch("questionary.confirm") as mock_confirm:
#        mock_confirm.return_value.ask.return_value = True
#
#        # Update second human
#        add("human", human_2, text_3)
#
#        # Expect to see update text
#        list("humans")
#        captured = capsys.readouterr()
#        output = captured.out[captured.out.find(human_1) :]
#
#        assert human_1 in output
#        assert text_1 in output
#        assert human_2 in output
#        assert output.count(human_2) == 1
#        assert text_3 in output
#        assert text_2 not in output
#
#    # Delete second human
#    delete("human", human_2)
#
#    # Expect second human to be deleted
#    list("humans")
#    captured = capsys.readouterr()
#    output = captured.out[captured.out.find(human_1) :]
#
#    assert human_1 in output
#    assert text_1 in output
#    assert human_2 not in output
#    assert text_2 not in output
#
#    # Clean up
#    delete("human", human_1)
#
#    reset_env_variables(server_url, token)
#
