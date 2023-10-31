---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Additional context**
Add any other context about the problem here.

---

**How did you install MemGPT?**

- From the official package? `pip install pymemgpt`
- From source?
  - With [poetry](https://github.com/cpacker/MemGPT#development)?
  - With `git clone git@github.com:cpacker/MemGPT.git` and `pip install -r requirements.txt`?

**Your setup (please complete the following information)**

- Your OS (Linux, MacOS, Windows)
- Where you're trying to run MemGPT from
  - Terminal? PowerShell? cmd.exe?
- Your python version (run `python --version`)
-  If you installed with `pip install pymemgpt`:
  - `pymemgpt` version (run `pip list | grep pymemgpt`)
- If you installed from source:
  - Your current git commit (run `git describe --tags`)

**Local LLM details**

If you are trying to run MemGPT with local LLMs, please provide the following information:

- The exact model you're trying to use (link to the HuggingFace page you downloaded it from)
  - We need the EXACT model name, including quantization scheme, e.g.: `dolphin-2.1-mistral-7b.Q6_K.gguf` (not just `dolphin-2.1-mistral`)
- The local LLM backend you are using (web UI? LM Studio?)
- Your hardware for the local LLM backend (local computer? operating system? remote RunPod?)
- Your hardware for the MemGPT command (same computer as the local LLM backend?)
- The full output (or as much as possible) of where the LLM is failing
  - If you can include screenshots, even better!
