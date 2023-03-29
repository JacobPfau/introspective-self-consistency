"""
This file will evaluate whether a model is able to choose the function which generated a sequence,
where we hold the correct function and sequence constant, and vary the incorrect options.

This allows us to evaluate whether the model is capable of recoginising a single correct function.
"""


import random
from typing import List, Tuple

from evals.question_eval import choose_function
from evals.wrong_int_func_generator import generate_wrong_functions


def capability_eval(
    target_sequence: str,
    prompt_file: str,
    model_name: str,
    temperature: float = 0.0,
    num_samples: int = 5,
    num_functions: int = 5,
    generate_functions: bool = False,
    incorrect_functions: List[str] = None,
    correct_functions: List[int] = None,
) -> Tuple[float, int]:
    """
    Given a sequence and some rule which could generate it, evaluate whether a model can discern
    the correct function that generated the sequence.
    We can either specify the list of possible functions, or specify some possible correct functions and generate
    different incorrect functions for each test.
    Returns the proportion of correct responses.
    """
    correct_choices = 0
    incorrect_choices = 0
    invalid_outputs = 0
    with open(prompt_file) as f:
        prompt = f.read()
        for i in range(num_samples):
            # Randomly choose one of the correct functions
            correct_function = random.choice(correct_functions)
            correct_function_index = random.randint(0, num_functions - 1)
            if generate_functions:
                # Generate incorrect functions
                incorrect_functions = generate_wrong_functions(
                    target_sequence, num_functions
                )
                sampled_functions = incorrect_functions
            else:
                # Randomly choose the incorrect functions
                incorrect_function_indices = random.sample(
                    range(num_functions), num_functions - 1
                )
                # Combine the correct and incorrect functions
                sampled_functions = [
                    incorrect_functions[i] for i in incorrect_function_indices
                ]
            sampled_functions.insert(correct_function_index, correct_function)
            # Choose the function
            model_response = choose_function(
                possible_functions=sampled_functions,
                correct_function_indices=[correct_function_index + 1],
                target_sequence=target_sequence,
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
            )
            if model_response == 1:
                correct_choices += 1
            elif model_response == 0:
                incorrect_choices += 1
            elif model_response < 0:
                invalid_outputs += 1
    return correct_choices, incorrect_choices, invalid_outputs


if __name__ == "__main__":
    # Test the capability eval
    target_sequence = "0,5,10,15"
    prompt_file = "evals/prompts/choose_function.txt"
    model_name = "DAVINCI"
    temperature = 0.0
    num_samples = 2
    num_functions = 4
    generate_functions = True
    incorrect_functions = [
        "lambda x: x",
        "lambda x: x + 7",
        "lambda x: x ** 2",
        "lambda x: x ** 3",
    ]
    correct_functions = ["lambda x: x*5"]

    correct_choices, incorrect_choices, invalid_outputs = capability_eval(
        target_sequence=target_sequence,
        prompt_file=prompt_file,
        model_name=model_name,
        temperature=temperature,
        num_samples=num_samples,
        num_functions=num_functions,
        generate_functions=generate_functions,
        incorrect_functions=incorrect_functions,
        correct_functions=correct_functions,
    )
    print(f"Correct choices: {correct_choices}")
    print(f"Incorrect choices: {incorrect_choices}")
    print(f"Invalid outputs: {invalid_outputs}")
