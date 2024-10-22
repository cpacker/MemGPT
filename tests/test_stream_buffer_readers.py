import json

import pytest

from letta.streaming_utils import JSONInnerThoughtsExtractor


@pytest.mark.parametrize("wait_for_first_key", [True, False])
def test_inner_thoughts_in_args_simple(wait_for_first_key):
    """Test case where the function_delta.arguments contains inner_thoughts

    Correct output should be inner_thoughts VALUE (not KEY) being written to one buffer
    And everything else (omiting inner_thoughts KEY) being written to the other buffer
    """
    print("Running Test Case 1: With 'inner_thoughts'")
    handler1 = JSONInnerThoughtsExtractor(inner_thoughts_key="inner_thoughts", wait_for_first_key=wait_for_first_key)
    fragments1 = [
        "{",
        """"inner_thoughts":"Chad's x2 tradition""",
        " is going strong! 😂 I love the enthusiasm!",
        " Time to delve into something imaginative:",
        """ If you could swap lives with any fictional character for a day, who would it be?\"""",
        ",",
        """"message":"Here we are again, with 'x2'!""",
        " 🎉 Let's take this chance: If you could swap",
        " lives with any fictional character for a day,",
        ''' who would it be?"''',
        "}",
    ]
    print("Basic inner thoughts testcase:", fragments1, "".join(fragments1))
    # Make sure the string is valid JSON
    _ = json.loads("".join(fragments1))

    if wait_for_first_key:
        # If we're waiting for the first key, then the first opening brace should be buffered/held back
        # until after the inner thoughts are finished
        expected_updates1 = [
            {"main_json_update": "", "inner_thoughts_update": ""},  # Fragment 1 (NOTE: different)
            {"main_json_update": "", "inner_thoughts_update": "Chad's x2 tradition"},  # Fragment 2
            {"main_json_update": "", "inner_thoughts_update": " is going strong! 😂 I love the enthusiasm!"},  # Fragment 3
            {"main_json_update": "", "inner_thoughts_update": " Time to delve into something imaginative:"},  # Fragment 4
            {
                "main_json_update": "",
                "inner_thoughts_update": " If you could swap lives with any fictional character for a day, who would it be?",
            },  # Fragment 5
            {"main_json_update": "", "inner_thoughts_update": ""},  # Fragment 6 (comma after inner_thoughts)
            {
                "main_json_update": '{"message":"Here we are again, with \'x2\'!',
                "inner_thoughts_update": "",
            },  # Fragment 7  (NOTE: the brace is included here, instead of at the beginning)
            {"main_json_update": " 🎉 Let's take this chance: If you could swap", "inner_thoughts_update": ""},  # Fragment 8
            {"main_json_update": " lives with any fictional character for a day,", "inner_thoughts_update": ""},  # Fragment 9
            {"main_json_update": ' who would it be?"', "inner_thoughts_update": ""},  # Fragment 10
            {"main_json_update": "}", "inner_thoughts_update": ""},  # Fragment 11
        ]
    else:
        # If we're not waiting for the first key, then the first opening brace should be written immediately
        expected_updates1 = [
            {"main_json_update": "{", "inner_thoughts_update": ""},  # Fragment 1
            {"main_json_update": "", "inner_thoughts_update": "Chad's x2 tradition"},  # Fragment 2
            {"main_json_update": "", "inner_thoughts_update": " is going strong! 😂 I love the enthusiasm!"},  # Fragment 3
            {"main_json_update": "", "inner_thoughts_update": " Time to delve into something imaginative:"},  # Fragment 4
            {
                "main_json_update": "",
                "inner_thoughts_update": " If you could swap lives with any fictional character for a day, who would it be?",
            },  # Fragment 5
            {"main_json_update": "", "inner_thoughts_update": ""},  # Fragment 6 (comma after inner_thoughts)
            {"main_json_update": '"message":"Here we are again, with \'x2\'!', "inner_thoughts_update": ""},  # Fragment 7
            {"main_json_update": " 🎉 Let's take this chance: If you could swap", "inner_thoughts_update": ""},  # Fragment 8
            {"main_json_update": " lives with any fictional character for a day,", "inner_thoughts_update": ""},  # Fragment 9
            {"main_json_update": ' who would it be?"', "inner_thoughts_update": ""},  # Fragment 10
            {"main_json_update": "}", "inner_thoughts_update": ""},  # Fragment 11
        ]

    for idx, (fragment, expected) in enumerate(zip(fragments1, expected_updates1)):
        updates_main_json, updates_inner_thoughts = handler1.process_fragment(fragment)
        # Assertions
        assert (
            updates_main_json == expected["main_json_update"]
        ), f"Test Case 1, Fragment {idx+1}: Main JSON update mismatch.\nExpected: '{expected['main_json_update']}'\nGot: '{updates_main_json}'"
        assert (
            updates_inner_thoughts == expected["inner_thoughts_update"]
        ), f"Test Case 1, Fragment {idx+1}: Inner Thoughts update mismatch.\nExpected: '{expected['inner_thoughts_update']}'\nGot: '{updates_inner_thoughts}'"


@pytest.mark.parametrize("wait_for_first_key", [True, False])
def test_inner_thoughts_in_args_trailing_quote(wait_for_first_key):
    # Another test case where there's a function call that has a chunk that ends with a double quote
    print("Running Test Case: chunk ends with double quote")
    handler1 = JSONInnerThoughtsExtractor(inner_thoughts_key="inner_thoughts", wait_for_first_key=wait_for_first_key)
    fragments1 = [
        # 1
        "{",
        # 2
        """\"inner_thoughts\":\"User wants to add 'banana' again for a fourth time; I'll track another addition.""",
        # 3
        '",',
        # 4
        """\"content\":\"banana""",
        # 5
        """\",\"""",
        # 6
        """request_heartbeat\":\"""",
        # 7
        """true\"""",
        # 8
        "}",
    ]
    print("Double quote test case:", fragments1, "".join(fragments1))
    # Make sure the string is valid JSON
    _ = json.loads("".join(fragments1))

    if wait_for_first_key:
        # If we're waiting for the first key, then the first opening brace should be buffered/held back
        # until after the inner thoughts are finished
        expected_updates1 = [
            {"main_json_update": "", "inner_thoughts_update": ""},  # Fragment 1 (NOTE: different)
            {
                "main_json_update": "",
                "inner_thoughts_update": "User wants to add 'banana' again for a fourth time; I'll track another addition.",
            },  # Fragment 2
            {"main_json_update": "", "inner_thoughts_update": ""},  # Fragment 3
            {
                "main_json_update": '{"content":"banana',
                "inner_thoughts_update": "",
            },  # Fragment 4
            {
                # "main_json_update": '","',
                "main_json_update": '",',
                "inner_thoughts_update": "",
            },  # Fragment 5
            {
                # "main_json_update": 'request_heartbeat":"',
                "main_json_update": '"request_heartbeat":"',
                "inner_thoughts_update": "",
            },  # Fragment 6
            {
                "main_json_update": 'true"',
                "inner_thoughts_update": "",
            },  # Fragment 7
            {
                "main_json_update": "}",
                "inner_thoughts_update": "",
            },  # Fragment 8
        ]
    else:
        pass
        # If we're not waiting for the first key, then the first opening brace should be written immediately
        expected_updates1 = [
            {"main_json_update": "{", "inner_thoughts_update": ""},  # Fragment 1 (NOTE: different)
            {
                "main_json_update": "",
                "inner_thoughts_update": "User wants to add 'banana' again for a fourth time; I'll track another addition.",
            },  # Fragment 2
            {"main_json_update": "", "inner_thoughts_update": ""},  # Fragment 3
            {
                "main_json_update": '"content":"banana',
                "inner_thoughts_update": "",
            },  # Fragment 4
            {
                # "main_json_update": '","',
                "main_json_update": '",',
                "inner_thoughts_update": "",
            },  # Fragment 5
            {
                # "main_json_update": 'request_heartbeat":"',
                "main_json_update": '"request_heartbeat":"',
                "inner_thoughts_update": "",
            },  # Fragment 6
            {
                "main_json_update": 'true"',
                "inner_thoughts_update": "",
            },  # Fragment 7
            {
                "main_json_update": "}",
                "inner_thoughts_update": "",
            },  # Fragment 8
        ]

    current_inner_thoughts = ""
    current_main_json = ""
    for idx, (fragment, expected) in enumerate(zip(fragments1, expected_updates1)):
        updates_main_json, updates_inner_thoughts = handler1.process_fragment(fragment)
        # Assertions
        assert (
            updates_main_json == expected["main_json_update"]
        ), f"Test Case 1, Fragment {idx+1}: Main JSON update mismatch.\nFragment: '{fragment}'\nExpected: '{expected['main_json_update']}'\nGot: '{updates_main_json}'\nCurrent JSON: '{current_main_json}'\nCurrent Inner Thoughts: '{current_inner_thoughts}'"
        assert (
            updates_inner_thoughts == expected["inner_thoughts_update"]
        ), f"Test Case 1, Fragment {idx+1}: Inner Thoughts update mismatch.\nExpected: '{expected['inner_thoughts_update']}'\nGot: '{updates_inner_thoughts}'\nCurrent JSON: '{current_main_json}'\nCurrent Inner Thoughts: '{current_inner_thoughts}'"
        current_main_json += updates_main_json
        current_inner_thoughts += updates_inner_thoughts

    print(f"Final JSON: '{current_main_json}'")
    print(f"Final Inner Thoughts: '{current_inner_thoughts}'")
    _ = json.loads(current_main_json)


def test_inner_thoughts_not_in_args():
    """Test case where the function_delta.arguments does not contain inner_thoughts

    Correct output should be everything being written to the main_json buffer
    """
    print("Running Test Case 2: Without 'inner_thoughts'")
    handler2 = JSONInnerThoughtsExtractor(inner_thoughts_key="inner_thoughts")
    fragments2 = [
        "{",
        """"message":"Here we are again, with 'x2'!""",
        " 🎉 Let's take this chance: If you could swap",
        " lives with any fictional character for a day,",
        ''' who would it be?"''',
        "}",
    ]
    print("Basic inner thoughts not in kwargs testcase:", fragments2, "".join(fragments2))
    # Make sure the string is valid JSON
    _ = json.loads("".join(fragments2))

    expected_updates2 = [
        {"main_json_update": "{", "inner_thoughts_update": ""},  # Fragment 1
        {"main_json_update": '"message":"Here we are again, with \'x2\'!', "inner_thoughts_update": ""},  # Fragment 2
        {"main_json_update": " 🎉 Let's take this chance: If you could swap", "inner_thoughts_update": ""},  # Fragment 3
        {"main_json_update": " lives with any fictional character for a day,", "inner_thoughts_update": ""},  # Fragment 4
        {"main_json_update": ' who would it be?"', "inner_thoughts_update": ""},  # Fragment 5
        {"main_json_update": "}", "inner_thoughts_update": ""},  # Fragment 6
    ]

    for idx, (fragment, expected) in enumerate(zip(fragments2, expected_updates2)):
        updates_main_json, updates_inner_thoughts = handler2.process_fragment(fragment)
        # Assertions
        assert (
            updates_main_json == expected["main_json_update"]
        ), f"Test Case 2, Fragment {idx+1}: Main JSON update mismatch.\nExpected: '{expected['main_json_update']}'\nGot: '{updates_main_json}'"
        assert (
            updates_inner_thoughts == expected["inner_thoughts_update"]
        ), f"Test Case 2, Fragment {idx+1}: Inner Thoughts update mismatch.\nExpected: '{expected['inner_thoughts_update']}'\nGot: '{updates_inner_thoughts}'"

    # Final assertions for Test Case 2
    expected_final_main_json2 = '{"message":"Here we are again, with \'x2\'! 🎉 Let\'s take this chance: If you could swap lives with any fictional character for a day, who would it be?"}'
    expected_final_inner_thoughts2 = ""

    assert (
        handler2.main_json == expected_final_main_json2
    ), f"Test Case 2: Final main_json mismatch.\nExpected: '{expected_final_main_json2}'\nGot: '{handler2.main_json}'"
    assert (
        handler2.inner_thoughts == expected_final_inner_thoughts2
    ), f"Test Case 2: Final inner_thoughts mismatch.\nExpected: '{expected_final_inner_thoughts2}'\nGot: '{handler2.inner_thoughts}'"
