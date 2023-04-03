"""
Create a prompt to select the correct function from a list of functions, given a target sequence.

Able to generate n_shot examples, and for each shot provide n_functions + 1 functions to choose from.

Will use some work from pipelines.sequence_completions.py

General format:
SYSTEM_PROMPT + (BASE_PROMPT + BASE_PROMPT_COMPLETION (including answer)
+ COT_PROMPT + COT_PROMPT_COMPLETION) * n_shot

"""
import random
from typing import Dict, List, Union

from evals.utils import _generate_random_function, generate_wrong_functions
from models.openai_model import CHAT_MODEL_NAME, DAVINCI_MODEL_NAME
from pipelines.sequence_completions import (  # BASE_PROMPT,; COT_PROMPT,; COT_STEP,
    SYSTEM_PROMPT,
    sequence_functions,
)


def function_selection_prompt(
    num_shots: int = 4,
    sequence_length: int = 5,
    num_functions: int = 4,
    use_cot: bool = False,
    model_name: str = DAVINCI_MODEL_NAME,
) -> Union[str, List[Dict[str, str]]]:
    """
    Create start of prompt for function selection task.
    This only includes the few shot examples
    """
    # TODO: refactor this to reduce duplication
    if model_name == CHAT_MODEL_NAME:
        prompt_turns = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
        ]
        for i in range(num_shots):
            question_text, function, sequence, correct_index = create_question_text(
                num_functions, use_cot, sequence_length
            )
            prompt_turns.append(
                {
                    "role": "user",
                    "content": question_text,
                }
            )
            if use_cot:
                cot_text = create_cot_text(function, sequence)
                prompt_turns.append(
                    {
                        "role": "user",
                        "content": cot_text,
                    }
                )
            # Add answer
            prompt_turns.append(
                {
                    "role": "user",
                    "content": f"{correct_index + 1}",
                }
            )
        prompt = prompt_turns
    elif model_name == DAVINCI_MODEL_NAME:
        prompt = SYSTEM_PROMPT + "\n"
        for i in range(num_shots):
            question_text, function, sequence, correct_index = create_question_text(
                num_functions, use_cot, sequence_length
            )
            prompt += question_text
            if use_cot:
                cot_text = create_cot_text(function, sequence)
                prompt += cot_text
            # Add answer
            prompt += f"A: {correct_index + 1}\n\n"
    # print(prompt)
    return prompt


def create_question_text(
    num_functions: int, use_cot: bool, sequence_length: int
) -> str:
    """
    Create the question text for a single shot.
    This will look like:
        Which python function generated the following sequence?
        0,1,2
        1. lambda x: x
        2. lambda x: x ** 420
        3. lambda x: 2 ** x
        4. lambda x: 100 // x
        A: 1
    """
    output = "Which python function generated the following sequence?\n"
    # Generate a target function
    function, offset = _generate_random_function(sequence_functions, (0, 10), (0, 10))
    # Generate a sequence from the function
    sequence = [eval(function)(i + offset) for i in range(sequence_length)]
    # Generate incorrect functions
    incorrect_functions = generate_wrong_functions(sequence, num_functions)
    # Combine all functions
    all_functions = incorrect_functions + [function]
    # Shuffle the functions
    random.shuffle(all_functions)
    # Get the index of the correct function
    correct_index = all_functions.index(function)
    # Add the sequence to the output
    output += ",".join([str(i) for i in sequence]) + "\n"
    # Add the incorrect functions to the output
    for i, fn in enumerate(all_functions):
        output += f"{i + 1}. {fn}\n"
    return output, function, sequence, correct_index


def create_cot_text(function: str, sequence: list[int]) -> str:
    """
    Create the COT text for a single shot.
    """
    # TODO: Implement COT (wait on Dom response)
