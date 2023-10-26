import os
import re
import tiktoken
import json

# Define the directory where the documentation resides
docs_dir = "text"

encoding = tiktoken.encoding_for_model("gpt-4")
PASSAGE_TOKEN_LEN = 800


def extract_text_from_sphinx_txt(file_path):
    lines = []
    title = ""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if not title:
                title = line.strip()
                continue
            if line and re.match(r"^.*\S.*$", line) and not re.match(r"^[-=*]+$", line):
                lines.append(line)
    passages = []
    curr_passage = []
    curr_token_ct = 0
    for line in lines:
        try:
            line_token_ct = len(encoding.encode(line, allowed_special={"<|endoftext|>"}))
        except Exception as e:
            print("line", line)
            raise e
        if line_token_ct > PASSAGE_TOKEN_LEN:
            passages.append(
                {
                    "title": title,
                    "text": line[:3200],
                    "num_tokens": curr_token_ct,
                }
            )
            continue
        curr_token_ct += line_token_ct
        curr_passage.append(line)
        if curr_token_ct > PASSAGE_TOKEN_LEN:
            passages.append({"title": title, "text": "".join(curr_passage), "num_tokens": curr_token_ct})
            curr_passage = []
            curr_token_ct = 0

    if len(curr_passage) > 0:
        passages.append({"title": title, "text": "".join(curr_passage), "num_tokens": curr_token_ct})
    return passages


# Iterate over all files in the directory and its subdirectories
passages = []
total_files = 0
for subdir, _, files in os.walk(docs_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(subdir, file)
            passages.append(extract_text_from_sphinx_txt(file_path))
            total_files += 1
print("total .txt files:", total_files)

# Save to a new text file or process as needed
with open("all_docs.jsonl", "w", encoding="utf-8") as file:
    for p in passages:
        file.write(json.dumps(p))
        file.write("\n")
