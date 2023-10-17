import random
import re
from typing import Dict, List, Tuple, Union

from src.models.openai_model import (
    DAVINCI_MODEL_NAME,
    generate_chat_completion,
    generate_text_completion,
)
from src.pipelines.baseb_sequence_completions import numberToBase
from src.pipelines.sequence_completions import sequence_functions


def generate_wrong_functions(
    sequence: Union[str, List[int]],
    num_functions: int = 5,
    offset_range: Tuple[int, int] = (0, 10),
    num_range: Tuple[int, int] = (0, 10),
    func_pool: dict = sequence_functions,
) -> List[str]:
    """
    Given an integer sequence, and a method of generating functions, generate a list of incorrect functions.
    Uses the sequence_functions dictionary in pipelines.sequence_completions, with offsets
    """
    if isinstance(sequence, str):
        # Turn the sequence into a list of ints
        sequence = [int(x.strip()) for x in sequence.split(",")]
    sequence_length = len(sequence)
    output = []
    i = 0
    while i < len(range(num_functions)):
        fn, _ = _generate_random_function(func_pool, num_range, offset_range)
        # TODO: will just check no equivalence for the first ten possible offsets, might want to change this
        correct = False
        for offset in range(10):
            # Check that the candidate is incorrect
            for step in range(sequence_length):
                fn_step = eval(fn)(step + offset)
                if fn_step != sequence[step]:
                    break
                elif step == sequence_length - 1:
                    correct = True
                    break
            if correct:
                break
        if not correct:
            i += 1
            output.append(fn)

    return output


def _generate_random_function(
    func_pool: dict, num_range: Tuple[int, int], offset_range: Tuple[int, int]
) -> Tuple[str, int]:
    """
    Given a pool of functions, randomly sample one, and an offset.
    """
    fn = random.choice(list(func_pool.values()))
    # Incorporate two numbers into the function
    # TODO: refactor this, is kind of ugly
    fn = fn.format(
        random.choice(list(range(*num_range))), random.choice(list(range(*num_range)))
    )
    offset = random.choice(list(range(offset_range[0], offset_range[1])))
    return (fn, offset)


def reformat_function(fn: str, offset: int) -> str:
    """
    Reformat a function to incorporate an offset, so the function is zero indexed.
    """
    first_occurrence = fn.find("x")
    replacement = f"(x + {offset})"
    if first_occurrence != -1:
        fn = fn[:first_occurrence] + "<placeholder>" + fn[first_occurrence + len("x") :]

    # replace all occurrences of x
    fn = fn.replace("x", replacement)
    # restore the first occurrence
    fn = fn.replace("<placeholder>", "x", 1)

    return fn


def format_question(
    prompt: Union[str, List[Dict[str, str]]],
    target_sequence: str,
    functions: List[str],
    model_name: str = "DAVINCI",
    base: int = 10,
) -> str:
    """
    Take the given few-shot prompt, add the question for the sequence, and the list of functions.
    """
    formatted_answers = "\n".join(
        [f"{i+1}. {func}" for i, func in enumerate(functions)]
    )
    question_text = f"""Which python function generated the following sequence?
Note that the sequence is now represented in base-{base}.\n{target_sequence}\n{formatted_answers}\n"""
    if model_name == "DAVINCI":
        result = f"{prompt}\n{question_text}"
    elif model_name == "CHAT":
        result = prompt
        result.append({"role": "user", "content": question_text})
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return result


def parse_model_response(model_response: str) -> int:
    """
    Parse the model's response to get the index of the function it chose.
    """
    # print("model response is: ", model_response)
    model_response = model_response.strip()
    if model_response == "":
        raise ValueError("Model response is empty")
    # If the answer isn't correctly formatted, report an error
    model_response = model_response.split("\n .,")
    if len(model_response) > 1:
        raise ValueError("More than one answer provided")
    else:
        model_response = model_response[0]
    # If the answer isn't a number, report an error
    try:
        model_response = int(model_response)
    except ValueError:
        raise ValueError("Not valid integer response")
    else:
        return model_response


def choose_function(
    possible_functions: List[str],
    correct_function_indices: List[int],
    target_sequence: str,
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
    base: int = 10,
) -> int:
    """
    Prompt a model to chose a function, from a list of possible functions, that generated a target sequence.
    The prompt should provide instructions for the task. We will edit the prompt to include the possible functions.
    Compare the model's choice to the correct function (which is also provided).
    Assume for now that the model is an openai model, either DAVINCI or CHAT.

    Returns 1 if the model's response is correct, 0 if the model's response is incorrect.
    Raises a ValueError if the model's response is invalid (wrong format, or empty).
    """
    # First, format the prompt to include the possible functions
    formatted_prompt = format_question(
        prompt=prompt,
        target_sequence=target_sequence,
        functions=possible_functions,
        model_name=model_name,
        base=base,
    )
    if model_name == "text-davinci-003":

        # Feed this into the model
        model_response = generate_text_completion(
            prompt=formatted_prompt,
            temperature=temperature,
            max_tokens=256,
            model=DAVINCI_MODEL_NAME,
        )
    elif model_name == "gpt-3.5-turbo-0301":
        # Feed this into the model
        model_response = generate_chat_completion(
            prompt_turns=formatted_prompt,
            temperature=temperature,
            max_tokens=256,
            model=model_name,
        )
        # print("model response is: ", model_response)
    # Parse the model's response to get the index of the function it chose
    try:
        model_response = parse_model_response(model_response)
    except ValueError as e:
        raise e

    # If the model's response is valid, compare it to the correct function
    else:
        print(model_response, correct_function_indices)
        # Compare the model's response to the correct function
        if model_response in correct_function_indices:
            return 1
        # If the model's response is incorrect, return 0
        else:
            return 0


def identify_function_class(fn: str) -> str:
    """
    Given a function, identify its class.
    Will do this using regex.
    """
    # Courtesy of GPT-4, may want to change this
    sequence_functions_regex = {
        "arithmetic_progression": re.compile(
            r"lambda\s*x:\s*\(\s*([\d\w]+)\s*\*\s*x\s*\)\s*\+\s*([\d\w]+)"
        ),
        "geometric_progression": re.compile(
            r"lambda\s*x:\s*\(\s*([\d\w]+)\s*\*\s*x\s*\)\s*\*\s*([\d\w]+)"
        ),
        "exponential_progression": re.compile(
            r"lambda\s*x:\s*\(\s*([\d\w]+)\s*\*\s*x\s*\)\s*\*\*\s*([\d\w]+)"
        ),
        "power_progression": re.compile(
            r"lambda\s*x:\s*([\d\w]+)\s*\*\*\s*\(\s*([\d\w]+)\s*\*\s*x\s*\)"
        ),
        "bit_or_progression": re.compile(
            r"lambda\s*x:\s*\(\s*([\d\w]+)\s*\*\s*x\s*\)\s*\|\s*([\d\w]+)"
        ),
        "modular_progression": re.compile(
            r"lambda\s*x:\s*\(x\s*\*\s*([\d\w]+)\)\s*%\s*\(\s*([\d\w]+)\s*\+\s*1\s*\)"
        ),
        "indexing_criteria_progression": re.compile(
            r"""\s*lambda\s+x\s*:\s*\[i\s+for\s+i\s+in\s+range\(\s*100\s*\)\s+if\s+i\s*%\s*
            \(\s*\d+\s*\+\s*1\s*\)\s+or\s+i\s*%\s*\(\s*\d+\s*\+\s*1\s*\)\]\s*\[\s*x\s*\]\s*"""
        ),
        "recursive_progression": re.compile(
            r"""\(lambda\s*a:lambda\s*v:a\(a,v\)\)\(lambda\s*fn,x:1\s*if\s*x==0\s*else\s*([\d\w]+)\s*\*\s*x\s*\*\s*
            fn\(fn,x-1\)\s*\+\s*([\d\w]+)\)"""
        ),
    }

    for function_type, regex_pattern in sequence_functions_regex.items():
        if regex_pattern.match(fn):
            return function_type
    return "Unknown"


def reformat_results(results: dict) -> dict:
    """
    Take the raw dictionary, reformat it so it's of the form:
    {
        "function_type_1": {
            "results":
                "fn_1": {
                    "correct": int,
                    "incorrect": int,
                },
                ...
            },
            "average accuracy": float,
            "total": int,
        },
        ...
    }

    Where total is the sum of correct and incorrect for that function type.
    """

    # First, we need to identify the function types
    function_types = set()
    for fn in results.keys():
        function_types.add(identify_function_class(fn))
    # Now, we can reformat the results
    reformatted_results = {}
    for function_type in function_types:
        reformatted_results[function_type] = {"results": {}, "total": 0}
        for fn in results.keys():
            if identify_function_class(fn) == function_type:
                reformatted_results[function_type]["results"][fn] = results[fn]
                # Change total to be in terms of number of successful examples
                reformatted_results[function_type]["total"] += (
                    results[fn]["correct"] + results[fn]["incorrect"]
                )
    # Now, we can calculate the average accuracy
    for function_type in reformatted_results.keys():
        total_correct = 0
        total_incorrect = 0
        for fn in reformatted_results[function_type]["results"].keys():
            total_correct += reformatted_results[function_type]["results"][fn][
                "correct"
            ]
            total_incorrect += reformatted_results[function_type]["results"][fn][
                "incorrect"
            ]
        reformatted_results[function_type]["average accuracy"] = total_correct / (
            total_correct + total_incorrect
        )
    return reformatted_results


def convert_numbers_to_base_b(string, base):
    """
    Convert all numbers in a string to base b.
    """

    def replace_number(match):
        number = int(match.group(0))
        return str(numberToBase(number, base))

    return re.sub(r"\d+", replace_number, string)


def reformat_ambiguous_sequences(ambiguous_sequences, base=2, max_length=30):
    """
    Reformat the keys of the ambiguous_sequences dictionary to be in an arbitrary base.

    Then, remove the ambiguous sequences that are too long.
    """

    reformatted_ambiguous_sequences = {}
    for ambiguous_sequence in ambiguous_sequences.keys():
        reformatted_ambiguous_sequence = convert_numbers_to_base_b(
            ambiguous_sequence, base
        )
        if len(reformatted_ambiguous_sequence) <= max_length:
            reformatted_ambiguous_sequences[
                reformatted_ambiguous_sequence
            ] = ambiguous_sequences[ambiguous_sequence]
    return reformatted_ambiguous_sequences
