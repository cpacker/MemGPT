SIMPLE = {
    "stopping_strings": [
        "\nUSER:",
        "\nASSISTANT:",
        "\nFUNCTION RETURN:",
        # '\n' +
        # '</s>',
        # '<|',
        # '\n#',
        # '\n\n\n',
    ],
    "max_new_tokens": 3072,
    # "truncation_length": 4096,  # assuming llama2 models
    "truncation_length": 8000,  # assuming mistral 7b
}
