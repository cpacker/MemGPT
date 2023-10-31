# from unittest.mock import patch
import pexpect
import memgpt


# def test_your_cli_function():
#     with patch('questionary.prompt', return_value={'your_question_key': 'fake_answer'}):
#         result = your_cli_app.your_function_that_uses_questionary()
#         assert result == 'expected_result'


def test_cli_sequence():
    # Start the CLI process
    child = pexpect.spawn("memgpt --first")

    # Expect a prompt or some output to know when to send the next command
    child.expect("SomeExpectedOutputOrPromptRegex")

    # Send the '/save' command
    child.sendline("/save")

    # Optionally, you can add more expects and sends here, depending on the interactions
    # child.expect("ExpectedOutputAfterSaveRegex")

    # Additional asserts can go here based on the outputs/behavior
    # Finally, make sure to gracefully exit the application if needed
    child.sendline("exit")  # Replace with your CLI's exit command

    child.close()


if __name__ == "__main__":
    test_cli_sequence()
