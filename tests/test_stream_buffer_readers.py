import pytest

from letta.server.rest_api.interface import JSONInnerThoughtsExtractor

# TODO make the wait_for_first_key automatically run both versions in pytest


@pytest.mark.parametrize("wait_for_first_key", [True, False])
def test_inner_thoughts_in_args(wait_for_first_key):
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
        """ If you could swap lives with any fictional character for a day, who would it be?",""",
        ",",
        """"message":"Here we are again, with 'x2'!""",
        " 🎉 Let's take this chance: If you could swap",
        " lives with any fictional character for a day,",
        ''' who would it be?"''',
        "}",
    ]

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
