---
title: Troubleshooting
excerpt: FAQ for MemGPT + custom LLM backends
category: 6580da9a40bb410016b8b0c3
---

## Problems getting MemGPT + local LLMs set up

### "Unable to connect to host ...", "API call got non-200 response code"

This error happens when MemGPT tries to run the LLM on the remote server you specified, but the server isn't working as expected.

For example, this error can happen when you have a typo in your endpoint (notice the duplicate `/v1` in the URL):

```text
Exception: API call got non-200 response code (code=400, msg={"error": {"message": "Missing required input", "code": 400, "type": "InvalidRequestError", "param": "context"}}) for address: http://localhost:5001/v1/api/v1/generate. Make sure that the web UI server is running and reachable at http://localhost:5001/v1/api/v1/generate.
```

Correcting the endpoint from `http://localhost:5001/v1` to `http://localhost:5001` (no `/v1` suffix) fixes the example error.

## Common errors while running MemGPT with local LLMs

### "Warning: no wrapper specified for local LLM, using the default wrapper"

**You can ignore this warning.**

This warning means that you did not specify a specific wrapper using the `--model-wrapper` flag, so MemGPT is using the default wrapper. If you would like to silence this warning, specify a wrapper with `--model-wrapper` or during `memgpt configure`.

### "Failed to parse JSON from local LLM response"

This error occurs when the LLM you're using outputs a string that cannot be parsed into a MemGPT function call. This is basically an LLM output error - the LLM was not able to properly follow MemGPT instructions and generate a MemGPT-compatible function call string.

**You can reduce the frequency of these errors by using better models, and wrappers with grammar-based sampling**. For example, moving from a 2B model to a 70B model, or moving from a quantized model to the full unquantized version of the same model.

**If you use really small models (< 7B) or heavily quantized models (< Q5), you are likely to run into many MemGPT LLM output errors.** Try using the [recommended models first](local_llm) before experimenting with your own custom models.

Many JSON-related output errors can be fixed by using a wrapper that uses grammars (required a grammar-enabled backend). See instructions about [grammars here](local_llm).

For example, let's look at the following error:

```text
Failed to parse JSON from local LLM response - error: Failed to decode JSON from LLM output:
{
  "function": "send_message",
  "params": {
    "inner_thoughts": "Oops, I got their name wrong! I should apologize and correct myself.",
    "message": "Sorry about that! I assumed you were Chad. Welcome, Brad! "
 <|> - error
JSONDecodeError.init() missing 2 required positional arguments: 'doc' and 'pos'
```

In this example, the error is saying that the local LLM output the following string:

```text
{
  "function": "send_message",
  "params": {
    "inner_thoughts": "Oops, I got their name wrong! I should apologize and correct myself.",
    "message": "Sorry about that! I assumed you were Chad. Welcome, Brad! "
 <|>
```

This string is not correct JSON - it is missing closing brackets and has a stray "<|>". Correct JSON would look like this:

```json
{
  "function": "send_message",
  "params": {
    "inner_thoughts": "Oops, I got their name wrong! I should apologize and correct myself.",
    "message": "Sorry about that! I assumed you were Chad. Welcome, Brad! "
  }
}
```

### "Got back an empty response string from ..."

MemGPT asked the server to run the LLM, but got back an empty response. Double-check that your server is running properly and has context length set correctly (it should be set to 8k if using Mistral 7B models).

### "Unable to connect to endpoint" using Windows + WSL

>⚠️ We recommend using Anaconda Shell, as WSL has been known to have issues passing network traffic between WSL and the Windows host.
> Check the [WSL Issue Thread](https://github.com/microsoft/WSL/issues/5211) for more info.

If you still would like to try WSL, you must be on WSL version 2.0.5 or above with the installation from the Microsoft Store app.
You will need to verify your WSL network mode is set to "mirrored"

You can do this by checking the `.wslconfig` file in `%USERPROFILE%'

Add the following if the file does not contain:
```
[wsl2]
networkingMode=mirrored # add this line if the wsl2 section already exists
```

