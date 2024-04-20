# see https://lite.koboldai.net/koboldcpp_api#/v1/post_v1_generate
SIMPLE = {
    "stop_sequence": [
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
    # "max_context_length": LLM_MAX_TOKENS,
    "max_length": 512,
}
