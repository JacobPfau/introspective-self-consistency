import json
from pprint import pprint

import evals.utils as utils
from evals.prompts.choose_function import (
    convert_numbers_to_base_b,
    create_question_text,
    function_selection_prompt,
)
from models.openai_model import DAVINCI_MODEL_NAME  # , CHAT_MODEL_NAME

# TODO: make these proper tests


def test_generate_wrong_functions():
    sequence = "1, 2, 3, 4, 5"
    output = utils.generate_wrong_functions(sequence)
    print(output)


def test_choose_function():
    with open("evals/prompts/choose_function.txt") as f:
        prompt = f.read()
        possible_functions = [
            "lambda x: 2 * x",
            "lambda x: 3 ** (4 * x)",
            "lambda x: 2 ** x",
            "lambda x: 21",
        ]
        correct_function_indices = [3]
        target_sequence = "1,2,4,8"
        model_name = "DAVINCI"
        try:
            result = utils.choose_function(
                possible_functions=possible_functions,
                correct_function_indices=correct_function_indices,
                target_sequence=target_sequence,
                prompt=prompt,
                model_name=model_name,
            )
        except ValueError as e:
            print(e)
        else:
            print(result)


def prompt_test():
    prompt = function_selection_prompt(
        num_shots=5,
        sequence_length=5,
        num_functions=4,
        use_cot=False,
        model_name=DAVINCI_MODEL_NAME,
    )
    print(prompt)
    print("buggo")


def test_function_identification():
    input_string = "lambda x: (2 * x) + 3"
    classification = utils.identify_function_class(input_string)
    print(classification)


def test_result_reformatting():
    # Load the results
    with open(
        "evals/results/ambiguous_sequences_function_selection_evaluation/2023-04-12-17-26-45/results.json"
    ) as f:
        results = json.load(f)
    # Reformat the results
    pprint(results)
    reformatted_results = utils.reformat_results(results)
    pprint(reformatted_results)


def test_view_results():
    with open(
        "evals/results/ambiguous_sequences_function_selection_evaluation/2023-04-12-18-53-36/results.json"
    ) as f:
        results = json.load(f)
    pprint(results)


def test_indexing_criteria_regex():
    import re

    pattern = r"""\s*lambda\s+x\s*:\s*\[i\s+for\s+i\s+in\s+range\(\s*100\s*\)\s+if\s+i\s*%\s*\(\s*\d+
        \s*\+\s*1\s*\)\s+or\s+i\s*%\s*\(\s*\d+\s*\+\s*1\s*\)\]\s*\[\s*x\s*\]\s*"""
    string_to_check = (
        "lambda x: [i for i in range(100) if i % (7 + 1) or i % (3 + 1)][x]"
    )

    if re.fullmatch(pattern, string_to_check):
        print("The string matches the pattern")
    else:
        print("The string does not match the pattern")


def test_base_conversion():
    stringo = "lambda x: x ** 420 69 69 16 81"
    base = 10
    print(convert_numbers_to_base_b(stringo, base))


def test_create_question_text_baseb():
    # stringo = "lambda x: x ** 420 69 69 16 81"
    base = 2
    print(create_question_text(5, False, 6, base))


if __name__ == "__main__":
    test_base_conversion()
    test_create_question_text_baseb()
