"""
Create a prompt to continue a sequence of numbers, in an arbitrary base.

Prompts will take the form:
--------------------------

    Here are some examples of sequence continuations.
    Q: 2, 4, 6,
    A: 8

    Q: 1, 2, 3, 4, 5,
    A: 6

    Q: 9, 16, 25, 36
    A: 49

    ***CONTINUATION_PROMPT***

    A:
--------------------------
    For (text) explanations, this will be of the form:
--------------------------
    Here are some examples of sequence explanations, i.e. python functions
    which could have generated the preceding sequences, with associated offset.
    Sequence: 2, 4, 6
    Explanation: lambda x: 2*x
    Offset: 0

    Sequence: 1, 2, 3, 4, 5
    Explanation: lambda x: x
    Offset: 1

    Sequence: 9, 16, 25, 36
    Explanation: lambda x: x**2
    Offset: 3

    ***EXPLANATION_PROMPT***

    Explanation:


--------------------------

The sequences will be taken from the list of ambiguous sequences.

"""

# import random
from typing import List, Union

from evals.utils import _generate_random_function

# from evals.utils import _generate_random_function, generate_wrong_functions
from pipelines.sequence_completions import sequence_functions
from q11.prompts.distributions import DISTRIBUTIONS

# from pipelines.sequence_completions import (  # BASE_PROMPT,; COT_PROMPT,; COT_STEP,
#     SYSTEM_PROMPT,
#     sequence_functions,
# )


def create_continuation_prompt(
    sequence: List[int],
    distribution: str,
    model_name: str = "DAVINCI",
    base: int = 10,
    shots: int = 0,
    shot_method: str = "random",
) -> Union[str, List[dict]]:
    """
    Create a prompt to continue a sequence of numbers.
    """
    sequence_length = len(sequence)
    prompt_text = "" if model_name == "DAVINCI" else []
    if shots > 0:
        for i in range(shots):
            # Note: we are using the sequence length implicitly specified by
            # the target sequence to generate the prompts.
            shot_prompt = generate_cont_shot_prompt(
                shot_method, sequence_length, model_name, base
            )
            prompt_text += shot_prompt

    # TODO: Need to fix!!

    text = DISTRIBUTIONS[distribution]["continuation"]
    text += "\n"
    text += f"The sequence is in base {base}."
    text += "\nQ: "
    if base == 10:
        text += ",".join([str(x) for x in sequence])
    elif base == 2:
        text += ",".join([bin(x) for x in sequence])
    else:
        raise ValueError(f"Invalid base: {base}")
    if model_name == "DAVINCI":
        # Prepend to the shots
        pretext = "Here are some examples of sequence continuations."
        pretext += "\n"
        text = pretext + prompt_text + text
        text += "\n"
        text += "A: "
        return text
    elif model_name == "CHAT":
        pretext = [
            {
                "role": "system",
                "content": "Here are some examples of sequence continuations.",
            }
        ]
        whole_prompt = pretext + prompt_text + [{"role": "user", "content": text}]
        return whole_prompt
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def generate_cont_shot_prompt(
    shot_method, sequence_length, model_name="DAVINCI", base=10
):
    """
    Generate a single shot prompt for a continuation.
    """
    if shot_method == "random":
        fn, offset = _generate_random_function(sequence_functions, (0, 5), (0, 5))
        sequence = [eval(fn)(x + offset) for x in range(sequence_length)]
    else:
        raise ValueError(f"Invalid shot method: {shot_method}")

    if model_name == "DAVINCI":
        text = "Q: "
        if base == 10:
            text += ",".join([str(x) for x in sequence])
            a_text = str(eval(fn)(sequence_length + offset))
        elif base == 2:
            text += ",".join([bin(x) for x in sequence])
            a_text = bin(eval(fn)(sequence_length + offset))
        text += "\n"
        text += "A: "
        text += a_text
        text += "\n"
        return text

    elif model_name == "CHAT":
        if base == 10:
            q_text = ",".join([str(x) for x in sequence])
            a_text = str(eval(fn)(sequence_length + offset))
        elif base == 2:
            q_text = ",".join([bin(x) for x in sequence])
            a_text = bin(eval(fn)(sequence_length + offset))
        response = [{"role": "user", "content": q_text}]
        response += [{"role": "assistant", "content": a_text}]
        # print("responseo be: ", response)
        return response

    else:
        raise ValueError(f"Invalid model name: {model_name}")