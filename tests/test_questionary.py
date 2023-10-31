# from unittest.mock import patch
import pexpect
import time
import memgpt


# def test_your_cli_function():
#     with patch('questionary.prompt', return_value={'your_question_key': 'fake_answer'}):
#         result = your_cli_app.your_function_that_uses_questionary()
#         assert result == 'expected_result'


def test_cli_sequence():
    # Start the CLI process
    child = pexpect.spawn("memgpt --first")

    # Expect a prompt or some output to know when to send the next command
    child.expect("Continue with legacy CLI?")

    # Send 'Y' followed by newline
    child.sendline("Y")
    # time.sleep(1)  # Wait for a short while to let output be captured
    # print(child.before.decode() if child.before else "No output captured")

    # Since .memgpt is empty, should jump immediately to "Which model?"
    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.expect("Which model would you like to use?")
    child.sendline()

    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.expect("Which persona would you like MemGPT to use?")
    child.sendline()

    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.expect("Which user would you like to use?")
    child.sendline()

    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.expect("Would you like to preload anything into MemGPT's archival memory?")
    child.sendline("N")

    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.expect("Enter your message")
    child.sendline("/save")

    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.expect("Saved config file")
    child.sendline("/load")

    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.expect("Loaded persistence manager")
    child.sendline("/exit")

    child.expect("Finished.")

    # Send the '/save' command
    # child.sendline("/save")
    # time.sleep(1)  # Wait for a short while to let output be captured
    # print(child.before.decode() if child.before else "No output captured")

    # Optionally, you can add more expects and sends here, depending on the interactions
    # child.expect("ExpectedOutputAfterSaveRegex")

    # Additional asserts can go here based on the outputs/behavior
    # Finally, make sure to gracefully exit the application if needed
    # child.sendline("/exit")  # Replace with your CLI's exit command
    # time.sleep(1)  # Wait for a short while to let output be captured
    # print(child.before.decode() if child.before else "No output captured")

    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."


if __name__ == "__main__":
    test_cli_sequence()
