settings = {
    # "stopping_strings": [
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
        # airoboros specific
        "\n### ",
        # '\n' +
        # '</s>',
        # '<|',
        "\n#",
        # "\n\n\n",
        # prevent chaining function calls / multi json objects / run-on generations
        # NOTE: this requires the ability to patch the extra '}}' back into the prompt
        "  }\n}\n",
    ],
    # most lm frontends default to 0.7-0.8 these days
    # "temperature": 0.8,
}
