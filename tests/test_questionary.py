import pexpect


TIMEOUT = 30  # seconds


def test_legacy_cli_sequence():
    # Start the CLI process
    child = pexpect.spawn("memgpt --first --strip_ui")

    child.expect("Continue with legacy CLI?", timeout=TIMEOUT)
    # Send 'Y' followed by newline
    child.sendline("Y")

    # Since .memgpt is empty, should jump immediately to "Which model?"
    child.expect("Which model would you like to use?", timeout=TIMEOUT)
    child.sendline()

    child.expect("Which persona would you like MemGPT to use?", timeout=TIMEOUT)
    child.sendline()

    child.expect("Which user would you like to use?", timeout=TIMEOUT)
    child.sendline()

    child.expect("Would you like to preload anything into MemGPT's archival memory?", timeout=TIMEOUT)
    child.sendline()  # Default No

    child.expect("Testing messaging functionality", timeout=TIMEOUT)
    child.expect("Enter your message", timeout=TIMEOUT)
    child.sendline()  # Send empty message

    child.expect("Try again!", timeout=TIMEOUT)  # Empty message
    child.sendline("/save")

    child.expect("Saved checkpoint", timeout=TIMEOUT)
    child.sendline("/load")

    child.expect("Loaded persistence manager", timeout=TIMEOUT)
    child.sendline("/exit")
    child.expect("Finished.", timeout=TIMEOUT)

    child.expect(pexpect.EOF, timeout=TIMEOUT)  # Wait for child to exit
    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."


if __name__ == "__main__":
    test_legacy_cli_sequence()
