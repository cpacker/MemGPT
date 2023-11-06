### Problems getting MemGPT + local LLMs set up

#### The OPENAI_API_BASE environment variable is not defined. Please set it in your environment.

`OPENAI_API_BASE` is not set correctly. Set the variable using `export`, `set`, or `$Env:`, depending on your operating system.

#### The BACKEND_TYPE environment variable is not defined. Please set it in your environment.

`BACKEND_TYPE` is not set correctly. Set the variable using `export`, `set`, or `$Env:`, depending on your operating system.

#### "Provided OPENAI_API_BASE value (...) must begin with http:// or https://"

This happens when you have a typo in `OPENAI_API_BASE`.

For example, don't do:
```sh
# This will trigger the error
export OPENAI_API_BASE=localhost:5000
```

Instead do:
```sh
export OPENAI_API_BASE=http://localhost:5000
```

#### "Unable to connect to host ...", "API call got non-200 response code"

This error happens when MemGPT tries to run the LLM on the remote server you specified, but the server isn't working as expected.

For example, this error can happen when you have a typo in your `OPENAI_API_BASE`:
```sh
# OPENAI_API_BASE here should NOT have the extra '/v1' at the end
export OPENAI_API_BASE=http://localhost:5001/v1
export BACKEND_TYPE=webui
```
```text
Exception: API call got non-200 response code (code=400, msg={"error": {"message": "Missing required input", "code": 400, "type": "InvalidRequestError", "param": "context"}}) for address: http://localhost:5001/v1/api/v1/generate.Make sure that the web UI server is running and reachable at http://localhost:5001/v1/api/v1/generate.
```

Correcting the typo fixes this example error:
```sh
# OPENAI_API_BASE here should NOT have the extra '/v1' at the end
export OPENAI_API_BASE=http://localhost:5001
```

### Common errors while running MemGPT with local LLMs

#### "Warning: no wrapper specified for local LLM, using the default wrapper (you can remove this warning by specifying the wrapper with --model)"

**You can ignore this warning.**

This warning means that you did not specify a specific wrapper using the `--model` flag, so MemGPT is using the default wrapper. If you would like to silence this warning, specify a wrapper with `--model`.

#### "Failed to parse JSON from local LLM response"

This error occurs when the LLM you're using outputs a string that cannot be parsed into a MemGPT function call. This is basically an LLM output error - the LLM was not able to properly follow MemGPT instructions and generate a MemGPT-compatible function call string.

**You can reduce the frequency of these errors by using better models, and wrappers with grammar-based sampling**. For example, moving from a 2B model to a 70B model, or moving from a quantized model to the full unquantized version of the same model.

**If you use really small models (< 7B) or heavily quantized models (< Q5), you are likely to run into many MemGPT LLM output errors.** Try using the [recommended models first](../local_llm) before experimenting with your own custom models.

Many JSON-related output errors can be fixed by using a wrapper that uses grammars (required a grammar-enabled backend). See instructions about [grammars here](../local_llm).

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

#### "Got back an empty response string from ..."

MemGPT asked the server to run the LLM, but got back an empty response. Double-check that your server is running properly and has context length set correctly (it should be set to 8k if using Mistral 7B models).
