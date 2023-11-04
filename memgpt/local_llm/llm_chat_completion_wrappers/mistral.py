# References - Mistral-7B With Function calling documentation
# https://huggingface.co/Trelis/Mistral-7B-Instruct-v0.1-function-calling-v2

import json
from .wrapper_base import LLMChatCompletionWrapper

# Define the roles and markers
B_INST, E_INST = "[INST]", "[/INST]"
B_FUNC, E_FUNC = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


class MistralChatCompletionWrapper(LLMChatCompletionWrapper):
    def chat_completion_to_prompt(self, messages, functions):
        print(" VVVVVVV")
        # print(messages)
        function_list = "\n".join(json.dumps(func, indent=4) for func in functions)
        # system_prompt = ' '.join(msg['content'] for msg in messages if msg['role'] == 'system')
        # user_prompt = ' '.join(msg['content'] for msg in messages if msg['role'] == 'user')

        # #if you don't do this the chat goes off the rails because has too many instructions
        # # Separate messages into system and user lists
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]

        # If there are user messages, separate the last user message from the others
        if user_messages:
            last_user_message = user_messages.pop()  # Remove and get the last user message
        else:
            last_user_message = ""

        # Now, join all the system messages and the user messages (except the last one) together for system_prompt
        system_prompt = " ".join(system_messages + user_messages)

        # And keep the last user message for user_prompt
        user_prompt = last_user_message

        # print(f"{B_FUNC}{function_list.strip()}{E_FUNC}{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n")
        print(type(user_prompt))
        return f"{B_FUNC}{function_list.strip()}{E_FUNC}{B_SYS}{system_prompt.strip()}{E_SYS}{B_INST}{user_prompt.strip()} {E_INST}\n\n"

    # def output_to_chat_completion_response(self, raw_llm_output):
    #     try:
    #         raw_output_json = json.loads(raw_llm_output)

    #         # print(raw_llm_output)
    #         print("raw_output_json")
    #         print(raw_output_json)
    #         response_content = raw_output_json
    #         response_json_start = response_content.find('{')
    #         response_json_end = response_content.rfind('}') + 1
    #         response_json = json.loads(response_content[response_json_start:response_json_end])
    #         return {
    #             'role': 'assistant',
    #             # 'content': response_content,
    #             'content': response_content[:response_json_start].strip(),
    #             'function': response_json.get('function'),
    #             'args': response_json.get('arguments', {})
    #         }
    #     except json.JSONDecodeError:
    #         return {
    #             'role': 'assistant',
    #             'content': raw_llm_output,
    #             'function': None,
    #             'args': {}
    #         }
    def output_to_chat_completion_response(self, raw_llm_output):
        try:
            # Parse raw_llm_output into a dictionary
            response_json = json.loads(raw_llm_output)
            print("Parsed JSON:")
            print(response_json)

            return {
                "role": "assistant",
                "content": response_json["message"],
                "function": response_json.get("function"),
                "args": response_json.get("arguments", {}),
            }
        except json.JSONDecodeError as e:
            # This block will catch and handle JSON decoding errors
            return {"role": "assistant", "content": f"Failed to parse JSON: {str(e)}", "function": None, "args": {}}
        except Exception as e:
            # This block will catch and handle all other exceptions
            return {"role": "assistant", "content": str(e), "function": None, "args": {}}
