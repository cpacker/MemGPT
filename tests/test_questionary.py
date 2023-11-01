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
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Continue with legacy CLI?")

    # Send 'Y' followed by newline
    child.sendline("Y")
    # time.sleep(1)  # Wait for a short while to let output be captured
    # print(child.before.decode() if child.before else "No output captured")

    # Since .memgpt is empty, should jump immediately to "Which model?"
    time.sleep(0.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Which model would you like to use?")
    child.sendline()

    time.sleep(0.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Which persona would you like MemGPT to use?")
    child.sendline()

    time.sleep(0.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Which user would you like to use?")
    child.sendline()

    time.sleep(0.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Would you like to preload anything into MemGPT's archival memory?")
    child.sendline("N")

    time.sleep(2.0)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Testing messaging functionality")
    # child.expect("Enter your message")
    child.sendline()
    child.sendline()
    child.sendline()

    time.sleep(1.0)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Try again!")
    # child.expect("Enter your message")
    child.sendline("/save")

    time.sleep(2.0)  # Wait for a short while to let output be captured
    print("(pre-save) DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("(pre-save) DEBUG AFTER:", child.after)
    try:
        child.expect("Saved checkpoint")  # erroring
    except:
        print("(post-save) DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
        print("(post-save) DEBUG AFTER:", child.after)
        raise
    child.sendline("/load")

    time.sleep(0.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Loaded persistence manager")
    child.sendline("/exit")
    child.sendline()

    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Finished.")

    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."


if __name__ == "__main__":
    test_cli_sequence()
