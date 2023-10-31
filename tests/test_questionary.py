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

    # Send 'Y' followed by newline
    child.sendline("Y")
    print(child.before.decode())  # after child.sendline or child.expect

    # Expect a prompt or some output to know when to send the next command
    # child.expect("SomeExpectedOutputOrPromptRegex")

    # Send the '/save' command
    child.sendline("/save")
    print(child.before.decode())  # after child.sendline or child.expect

    # Optionally, you can add more expects and sends here, depending on the interactions
    # child.expect("ExpectedOutputAfterSaveRegex")

    # Additional asserts can go here based on the outputs/behavior
    # Finally, make sure to gracefully exit the application if needed
    child.sendline("/exit")  # Replace with your CLI's exit command
    print(child.before.decode())  # after child.sendline or child.expect

    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."


if __name__ == "__main__":
    test_cli_sequence()
