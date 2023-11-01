import pexpect
import sys
import time


def test_cli_sequence():
    # Start the CLI process
    child = pexpect.spawn("memgpt --first --strip_ui")

    # Expect a prompt or some output to know when to send the next command
    time.sleep(1.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Continue with legacy CLI?")

    # Send 'Y' followed by newline
    child.sendline("Y")
    # time.sleep(1)  # Wait for a short while to let output be captured
    # print(child.before.decode() if child.before else "No output captured")

    # Since .memgpt is empty, should jump immediately to "Which model?"
    time.sleep(1.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Which model would you like to use?")
    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.sendline()

    time.sleep(1.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Which persona would you like MemGPT to use?")
    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.sendline()

    time.sleep(1.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Which user would you like to use?")
    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.sendline()

    time.sleep(1.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Would you like to preload anything into MemGPT's archival memory?")
    time.sleep(0.5)  # Wait for a short while to let output be captured
    # child.sendline("N")
    child.sendline()

    time.sleep(2.0)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Testing messaging functionality")
    time.sleep(1.0)  # Wait for a short while to let output be captured
    # child.sendline()
    # child.sendline()
    sys.stdout.flush()
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Enter your message")
    child.sendline()

    time.sleep(1.0)  # Wait for a short while to let output be captured
    sys.stdout.flush()
    print("(pre-enter) DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("(pre-enter) DEBUG AFTER:", child.after)
    try:
        # child.expect(["Try again!", pexpect.EOF])
        child.expect("Try again!")
    except:
        print("(post-enter) DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
        print("(post-enter) DEBUG AFTER:", child.after)
        raise
    # child.expect("Enter your message")
    time.sleep(0.5)  # Wait for a short while to let output be captured
    sys.stdout.flush()
    child.sendline("/save")

    time.sleep(2.0)  # Wait for a short while to let output be captured
    print("(pre-save) DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("(pre-save) DEBUG AFTER:", child.after)
    try:
        # child.expect(["Saved checkpoint", pexpect.EOF])  # erroring
        child.expect("Saved checkpoint")  # erroring
    except:
        print("(post-save) DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
        print("(post-save) DEBUG AFTER:", child.after)
        raise
    time.sleep(0.5)  # Wait for a short while to let output be captured
    child.sendline("/load")

    time.sleep(1.5)  # Wait for a short while to let output be captured
    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    # child.expect(["Loaded persistence manager", pexpect.EOF])
    child.expect("Loaded persistence manager")
    time.sleep(0.5)  # Wait for a short while to let output be captured
    sys.stdout.flush()
    child.sendline("/exit")
    time.sleep(0.5)  # Wait for a short while to let output be captured
    sys.stdout.flush()
    child.sendline()

    print("DEBUG BEFORE:", child.before.decode() if child.before else "no child.before")
    print("DEBUG AFTER:", child.after)
    child.expect("Finished.")

    time.sleep(2.0)  # Wait for a short while to let output be captured
    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."


if __name__ == "__main__":
    test_cli_sequence()
