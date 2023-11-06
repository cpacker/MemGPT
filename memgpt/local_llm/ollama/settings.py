from ...constants import LLM_MAX_TOKENS

# see https://github.com/jmorganca/ollama/blob/main/docs/api.md
# and https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
SIMPLE = {
    "options": {
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
    },
    "stream": False,
    # turn off Ollama's own prompt formatting
    "system": None,
    "template": None,
    "context": None,
}
