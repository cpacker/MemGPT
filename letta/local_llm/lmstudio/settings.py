SIMPLE = {
    "stop": [
        "\nUSER:",
        "\nASSISTANT:",
        "\nFUNCTION RETURN:",
        "\nUSER",
        "\nASSISTANT",
        "\nFUNCTION RETURN",
        "\nFUNCTION",
        "\nFUNC",
        "<|im_start|>",
        "<|im_end|>",
        "<|im_sep|>",
        # '\n' +
        # '</s>',
        # '<|',
        # '\n#',
        # '\n\n\n',
    ],
    # This controls the maximum number of tokens that the model can generate
    # Cap this at the model context length (assuming 8k for Mistral 7B)
    # "max_tokens": 8000,
    # "max_tokens": LLM_MAX_TOKENS,
    # This controls how LM studio handles context overflow
    # In Letta we handle this ourselves, so this should be commented out
    # "lmstudio": {"context_overflow_policy": 2},
    "stream": False,
    "model": "local model",
}
