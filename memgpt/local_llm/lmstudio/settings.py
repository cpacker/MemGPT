SIMPLE = {
    "stop": [
        "\nUSER:",
        "\nASSISTANT:",
        "\nFUNCTION RETURN:",
        # '\n' +
        # '</s>',
        # '<|',
        # '\n#',
        # '\n\n\n',
    ],
    # This controls the maximum number of tokens that the model can generate
    # Cap this at the model context length (assuming 8k for Mistral 7B)
    "max_tokens": 8000,
    # This controls how LM studio handles context overflow
    # In MemGPT we handle this ourselves, so this should be commented out
    # "lmstudio": {"context_overflow_policy": 2},
    "stream": False,
    "model": "local model",
}
