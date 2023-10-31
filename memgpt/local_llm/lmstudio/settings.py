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
    "max_tokens": 3072,
    "lmstudio": {"context_overflow_policy": 2},
    "stream": False,
    "model": "local model",
}
