"""
The first function in this file (function_class_selection_evaluation) will evaluate
whether a model is able to choose the function which generated a sequence,
where the sequence is generated by a function from a given class. This allows us to
evaluate whether the model is capable of understanding a given function class.


The second function in this file (function_selection_evaluation) will evaluate whether a
model is able to choose the function which generated a sequence, where we hold the
correct function and sequence constant, and vary the incorrect options. This allows
us to evaluate whether the model is capable of recoginising a single correct function.
"""

import random
from copy import deepcopy
from typing import List, Tuple, Union

from evals.utils import choose_function, generate_wrong_functions
from models.openai_model import CHAT_MODEL_NAME, DAVINCI_MODEL_NAME
from src.prompt_generation.robustness_checks.choose_function import (
    function_selection_prompt,
)


def function_class_selection_evaluation(
    model_name: str,
    function_class: str,
    sequence_functions: dict[str, str],
    sequence_length: int,
    temperature: float = 0.0,
    num_shots: int = 4,
    num_samples: int = 50,
    num_functions: int = 5,
) -> Tuple[int, int, int]:
    """
    Given a class of function and a model, produce a number of sequences and ask the model to choose the function which
    generated the sequence. Return the number of correct responses.
    """
    correct = 0
    incorrect = 0
    invalid = 0
    fn_form = sequence_functions[function_class]
    # Generate a prompt
    prompt = function_selection_prompt(
        num_shots=num_shots,
        num_functions=num_functions,
        model_name=model_name,
    )
    for i in range(num_samples):
        print("Question: ", i + 1, "/", num_samples, sep="")
        # Generate a function from the class
        target_fn = fn_form.format(random.randint(0, 10), random.randint(0, 10))

        offset = random.randint(0, 10)
        # Generate a sequence
        target_sequence = [eval(target_fn)(j + offset) for j in range(sequence_length)]

        # Generate incorrect functions
        incorrect_functions = generate_wrong_functions(target_sequence, num_functions)

        all_functions = incorrect_functions + [target_fn]
        # Shuffle the functions
        random.shuffle(all_functions)
        # Get the index of the correct function
        correct_function_index = all_functions.index(target_fn)

        # Load the prompt
        # Prompt the model to choose the correct function
        try:
            model_response = choose_function(
                possible_functions=all_functions,
                correct_function_indices=[correct_function_index + 1],
                target_sequence=target_sequence,
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
            )
        except ValueError:
            invalid += 1
            continue
        if model_response == 1:
            correct += 1
        elif model_response == 0:
            incorrect += 1
    return correct, incorrect, invalid


def function_selection_evaluation(
    model_name: str,
    target_sequence: str,
    temperature: float = 0.0,
    num_shots: int = 4,
    num_samples: int = 5,
    num_functions: int = 5,
    generate_functions: bool = False,
    incorrect_functions: List[str] = None,
    correct_functions: List[int] = None,
    offset: Union[int, None] = None,
    base: int = 10,
    number_format: str = "None",
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

    if model_name == "CHAT":
        # Generate a prompt
        prompt = function_selection_prompt(
            num_shots=num_shots,
            num_functions=num_functions,
            model_name=CHAT_MODEL_NAME,
            base=base,
            num_format=number_format,
        )
    elif model_name == "DAVINCI":
        # Generate a prompt
        prompt = function_selection_prompt(
            num_shots=num_shots,
            num_functions=num_functions,
            model_name=DAVINCI_MODEL_NAME,
            base=base,
            num_format=number_format,
        )
    else:
        raise ValueError("Model name not recognised")

    for i in range(num_samples):
        # Randomly choose one of the correct functions
        correct_function = random.choice(correct_functions)
        correct_function_index = random.randint(0, num_functions - 1)
        if generate_functions:
            # Generate incorrect functions
            incorrect_functions = generate_wrong_functions(
                target_sequence, num_functions - 1
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
        try:
            new_prompt = deepcopy(prompt)

            model_response = choose_function(
                possible_functions=sampled_functions,
                correct_function_indices=[correct_function_index + 1],
                target_sequence=target_sequence,
                prompt=new_prompt,
                model_name=model_name,
                temperature=temperature,
                base=base,
            )

        except ValueError:
            invalid_outputs += 1
            continue
        if model_response == 1:
            correct_choices += 1
        elif model_response == 0:
            incorrect_choices += 1
    return correct_choices, incorrect_choices, invalid_outputs
