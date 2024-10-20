from typing import Optional

from letta.constants import DEFAULT_MESSAGE_TOOL_KWARG


class JSONInnerThoughtsExtractor:
    """
    A class to process incoming JSON fragments and extract 'inner_thoughts' separately from the main JSON.

    This handler processes JSON fragments incrementally, parsing out the value associated with a specified key (default is 'inner_thoughts'). It maintains two separate buffers:

    - `main_json`: Accumulates the JSON data excluding the 'inner_thoughts' key-value pair.
    - `inner_thoughts`: Accumulates the value associated with the 'inner_thoughts' key.

    **Parameters:**

    - `inner_thoughts_key` (str): The key to extract from the JSON (default is 'inner_thoughts').
    - `wait_for_first_key` (bool): If `True`, holds back main JSON output until after the 'inner_thoughts' value is processed.

    **Functionality:**

    - **Stateful Parsing:** Maintains parsing state across fragments.
    - **String Handling:** Correctly processes strings, escape sequences, and quotation marks.
    - **Selective Extraction:** Identifies and extracts the value of the specified key.
    - **Fragment Processing:** Handles data that arrives in chunks.

    **Usage:**

    ```python
    extractor = JSONInnerThoughtsExtractor(wait_for_first_key=True)
    for fragment in fragments:
        updates_main_json, updates_inner_thoughts = extractor.process_fragment(fragment)
    ```

    """

    def __init__(self, inner_thoughts_key="inner_thoughts", wait_for_first_key=False):
        self.inner_thoughts_key = inner_thoughts_key
        self.wait_for_first_key = wait_for_first_key
        self.main_buffer = ""
        self.inner_thoughts_buffer = ""
        self.state = "start"  # Possible states: start, key, colon, value, comma_or_end, end
        self.in_string = False
        self.escaped = False
        self.current_key = ""
        self.is_inner_thoughts_value = False
        self.inner_thoughts_processed = False
        self.hold_main_json = wait_for_first_key
        self.main_json_held_buffer = ""

    def process_fragment(self, fragment):
        updates_main_json = ""
        updates_inner_thoughts = ""
        i = 0
        while i < len(fragment):
            c = fragment[i]
            if self.escaped:
                self.escaped = False
                if self.in_string:
                    if self.state == "key":
                        self.current_key += c
                    elif self.state == "value":
                        if self.is_inner_thoughts_value:
                            updates_inner_thoughts += c
                            self.inner_thoughts_buffer += c
                        else:
                            if self.hold_main_json:
                                self.main_json_held_buffer += c
                            else:
                                updates_main_json += c
                                self.main_buffer += c
                else:
                    if not self.is_inner_thoughts_value:
                        if self.hold_main_json:
                            self.main_json_held_buffer += c
                        else:
                            updates_main_json += c
                            self.main_buffer += c
            elif c == "\\":
                self.escaped = True
                if self.in_string:
                    if self.state == "key":
                        self.current_key += c
                    elif self.state == "value":
                        if self.is_inner_thoughts_value:
                            updates_inner_thoughts += c
                            self.inner_thoughts_buffer += c
                        else:
                            if self.hold_main_json:
                                self.main_json_held_buffer += c
                            else:
                                updates_main_json += c
                                self.main_buffer += c
                else:
                    if not self.is_inner_thoughts_value:
                        if self.hold_main_json:
                            self.main_json_held_buffer += c
                        else:
                            updates_main_json += c
                            self.main_buffer += c
            elif c == '"':
                if not self.escaped:
                    self.in_string = not self.in_string
                    if self.in_string:
                        if self.state in ["start", "comma_or_end"]:
                            self.state = "key"
                            self.current_key = ""
                            # Release held main_json when starting to process the next key
                            if self.wait_for_first_key and self.hold_main_json and self.inner_thoughts_processed:
                                updates_main_json += self.main_json_held_buffer
                                self.main_buffer += self.main_json_held_buffer
                                self.main_json_held_buffer = ""
                                self.hold_main_json = False
                    else:
                        if self.state == "key":
                            self.state = "colon"
                        elif self.state == "value":
                            # End of value
                            if self.is_inner_thoughts_value:
                                self.inner_thoughts_processed = True
                                # Do not release held main_json here
                            else:
                                if self.hold_main_json:
                                    self.main_json_held_buffer += '"'
                                else:
                                    updates_main_json += '"'
                                    self.main_buffer += '"'
                            self.state = "comma_or_end"
                else:
                    self.escaped = False
                    if self.in_string:
                        if self.state == "key":
                            self.current_key += '"'
                        elif self.state == "value":
                            if self.is_inner_thoughts_value:
                                updates_inner_thoughts += '"'
                                self.inner_thoughts_buffer += '"'
                            else:
                                if self.hold_main_json:
                                    self.main_json_held_buffer += '"'
                                else:
                                    updates_main_json += '"'
                                    self.main_buffer += '"'
            elif self.in_string:
                if self.state == "key":
                    self.current_key += c
                elif self.state == "value":
                    if self.is_inner_thoughts_value:
                        updates_inner_thoughts += c
                        self.inner_thoughts_buffer += c
                    else:
                        if self.hold_main_json:
                            self.main_json_held_buffer += c
                        else:
                            updates_main_json += c
                            self.main_buffer += c
            else:
                if c == ":" and self.state == "colon":
                    self.state = "value"
                    self.is_inner_thoughts_value = self.current_key == self.inner_thoughts_key
                    if self.is_inner_thoughts_value:
                        pass  # Do not include 'inner_thoughts' key in main_json
                    else:
                        key_colon = f'"{self.current_key}":'
                        if self.hold_main_json:
                            self.main_json_held_buffer += key_colon + '"'
                        else:
                            updates_main_json += key_colon + '"'
                            self.main_buffer += key_colon + '"'
                elif c == "," and self.state == "comma_or_end":
                    if self.is_inner_thoughts_value:
                        # Inner thoughts value ended
                        self.is_inner_thoughts_value = False
                        self.state = "start"
                        # Do not release held main_json here
                    else:
                        if self.hold_main_json:
                            self.main_json_held_buffer += c
                        else:
                            updates_main_json += c
                            self.main_buffer += c
                        self.state = "start"
                elif c == "{":
                    if not self.is_inner_thoughts_value:
                        if self.hold_main_json:
                            self.main_json_held_buffer += c
                        else:
                            updates_main_json += c
                            self.main_buffer += c
                elif c == "}":
                    self.state = "end"
                    if self.hold_main_json:
                        self.main_json_held_buffer += c
                    else:
                        updates_main_json += c
                        self.main_buffer += c
                else:
                    if self.state == "value":
                        if self.is_inner_thoughts_value:
                            updates_inner_thoughts += c
                            self.inner_thoughts_buffer += c
                        else:
                            if self.hold_main_json:
                                self.main_json_held_buffer += c
                            else:
                                updates_main_json += c
                                self.main_buffer += c
            i += 1

        return updates_main_json, updates_inner_thoughts

    @property
    def main_json(self):
        return self.main_buffer

    @property
    def inner_thoughts(self):
        return self.inner_thoughts_buffer


class FunctionArgumentsStreamHandler:
    """State machine that can process a stream of"""

    def __init__(self, json_key=DEFAULT_MESSAGE_TOOL_KWARG):
        self.json_key = json_key
        self.reset()

    def reset(self):
        self.in_message = False
        self.key_buffer = ""
        self.accumulating = False
        self.message_started = False

    def process_json_chunk(self, chunk: str) -> Optional[str]:
        """Process a chunk from the function arguments and return the plaintext version"""

        # Use strip to handle only leading and trailing whitespace in control structures
        if self.accumulating:
            clean_chunk = chunk.strip()
            if self.json_key in self.key_buffer:
                if ":" in clean_chunk:
                    self.in_message = True
                    self.accumulating = False
                    return None
            self.key_buffer += clean_chunk
            return None

        if self.in_message:
            if chunk.strip() == '"' and self.message_started:
                self.in_message = False
                self.message_started = False
                return None
            if not self.message_started and chunk.strip() == '"':
                self.message_started = True
                return None
            if self.message_started:
                if chunk.strip().endswith('"'):
                    self.in_message = False
                    return chunk.rstrip('"\n')
                return chunk

        if chunk.strip() == "{":
            self.key_buffer = ""
            self.accumulating = True
            return None
        if chunk.strip() == "}":
            self.in_message = False
            self.message_started = False
            return None
        return None
