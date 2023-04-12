import random
import re
from typing import List, Tuple, Union

from models.openai_model import (
    DAVINCI_MODEL_NAME,
    generate_chat_completion,
    generate_text_completion,
)
from pipelines.sequence_completions import sequence_functions


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


def format_question(
    prompt: str,
    target_sequence: str,
    functions: List[str],
) -> str:
    formatted_answers = "\n".join(
        [f"{i+1}. {func}" for i, func in enumerate(functions)]
    )
    result = f"{prompt}\n{target_sequence}\n{formatted_answers}\nA:"
    return result


def parse_model_response(model_response: str) -> int:
    """
    Parse the model's response to get the index of the function it chose.
    """
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
    )
    if model_name == "DAVINCI":

        # Feed this into the model
        model_response = generate_text_completion(
            prompt=formatted_prompt,
            temperature=temperature,
            max_tokens=256,
            model=DAVINCI_MODEL_NAME,
        )
    elif model_name == "CHAT":
        # Feed this into the model
        model_response = generate_chat_completion(
            # TODO: make this more general, to include multiple turns
            # for few shot examples
            prompt_turns=[{"text": formatted_prompt}],
            temperature=temperature,
            max_tokens=256,
            model=model_name,
        )
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
            r"""lambda\s*x:\s*\[i\s*for\s*i\s*in\s*range\s*\(100\)\s*if\s*i\s*%\s*\(\s*([\d\w]+)\s*\+\s*1\s*\)\s*
            or\s*i\s*%\s*\(\s*([\d\w]+)\s*\+\s*1\s*\)\]\s*\[x\]"""
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
    """

    # First, we need to identify the function types
    function_types = set()
    for fn in results.keys():
        function_types.add(identify_function_class(fn))
    # Now, we can reformat the results
    reformatted_results = {}
    for function_type in function_types:
        reformatted_results[function_type] = {
            "results": {},
            "average accuracy": 0.0,
            "total": 0,
        }
        for fn in results.keys():
            if identify_function_class(fn) == function_type:
                reformatted_results[function_type]["results"][fn] = {
                    "correct": results[fn]["correct"],
                    "incorrect": results[fn]["incorrect"],
                }
                reformatted_results[function_type]["average accuracy"] += results[fn][
                    "correct"
                ] / (results[fn]["correct"] + results[fn]["incorrect"])
                reformatted_results[function_type]["total"] += results[fn]["correct"]
                reformatted_results[function_type]["total"] += results[fn]["incorrect"]
        reformatted_results[function_type]["average accuracy"] /= reformatted_results[
            function_type
        ]["total"]
    return reformatted_results
