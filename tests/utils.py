import pexpect

from .constants import TIMEOUT


def configure_memgpt(enable_openai=True, enable_azure=False):
    child = pexpect.spawn("memgpt configure")

    child.expect("Do you want to enable MemGPT with Open AI?", timeout=TIMEOUT)
    if enable_openai:
        child.sendline("y")
    else:
        child.sendline("n")

    child.expect("Do you want to enable MemGPT with Azure?", timeout=TIMEOUT)
    if enable_azure:
        child.sendline("y")
    else:
        child.sendline("n")

    child.expect("Select default preset:", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select default persona:", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select default human:", timeout=TIMEOUT)
    child.sendline()

    child.expect("Select storage backend for archival data:", timeout=TIMEOUT)
    child.sendline()

    child.expect(pexpect.EOF, timeout=TIMEOUT)  # Wait for child to exit
    child.close()
    assert child.isalive() is False, "CLI should have terminated."
    assert child.exitstatus == 0, "CLI did not exit cleanly."
