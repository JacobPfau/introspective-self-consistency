"""
Create a prompt to continue a sequence of numbers, in an arbitrary base.

Prompts will take the form:
--------------------------
    Here are some examples of sequence explanations, i.e. python functions
    which could have generated the preceding sequences. Assume the first
    number was generated by f(0), the second by f(1), and so on.
    Sequence: 2, 4, 6
    Explanation: lambda x: 2*(x+1)

    Sequence: 1, 2, 3, 4, 5
    Explanation: lambda x: (x+1)

    Sequence: 9, 16, 25, 36
    Explanation: lambda x: (x+3)**2

    ***EXPLANATION_PROMPT***

    Explanation:

"""


from typing import List, Union

from src.evals.prompts.distribution_prompt import DISTRIBUTIONS
from src.evals.utils import _generate_random_function, reformat_function

# from evals.utils import _generate_random_function, generate_wrong_functions
from src.models.openai_model import DAVINCI_MODEL_NAME
from src.pipelines.sequence_completions import sequence_functions

PRE_PROMPT = """
Here are some examples of sequence explanations, i.e. python functions
which generated the preceding sequences base {}. Assume the first number was generated by f(0),
the second by f(1), and so on.
"""

# TODO: fix generating functions to include recursive progressions, an ok fix for now.
sequence_functions = sequence_functions.copy()


def create_explanation_prompt(
    sequence: List[int],
    distribution: str,
    model_name: str = DAVINCI_MODEL_NAME,
    base: int = 10,
    shots: int = 0,
    shot_method: str = "random",
) -> Union[str, List[dict]]:
    """
    Create a prompt to continue a sequence of numbers.
    """
    sequence_length = len(sequence)
    prompt_text = "" if model_name == "text-davinci-003" else []
    if shots > 0:
        for i in range(shots):
            # Note: we are using the sequence length implicitly specified by
            # the target sequence to generate the prompts.
            shot_prompt = generate_exp_shot_prompt(
                shot_method, sequence_length, model_name, base
            )
            prompt_text += shot_prompt

    # TODO: Need to fix!!

    text = DISTRIBUTIONS[distribution]["explanation"]
    text += "\n"
    text += f"The sequence is in base {base}."
    text += "\nQ: "
    if base == 10:
        text += ",".join([str(x) for x in sequence])
    elif base == 2:
        text += ",".join([bin(x) for x in sequence])
    pre_prompt = PRE_PROMPT
    pre_prompt = pre_prompt.format(base)
    # print(pre_prompt)
    if model_name == "text-davinci-003":
        # Prepend to the shots
        pretext = pre_prompt + "\n"
        pretext += "\n"
        text = pretext + prompt_text + text
        text += "\n"
        return text
    elif model_name == "gpt-3.5-turbo":
        pretext = [
            {
                "role": "system",
                "content": pre_prompt,
            }
        ]
        whole_prompt = pretext + prompt_text + [{"role": "user", "content": text}]
        print(whole_prompt)
        return whole_prompt
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def generate_exp_shot_prompt(
    shot_method, sequence_length, model_name=DAVINCI_MODEL_NAME, base=10
):
    """
    Generate a single shot prompt for a explanation.
    """
    if shot_method == "random":
        fn, offset = _generate_random_function(sequence_functions, (0, 7), (0, 7))
        # Reformat fn to replace every x after the first with x+offset
        fn = reformat_function(fn, offset)
        sequence = [eval(fn)(x) for x in range(sequence_length)]
    else:
        raise ValueError(f"Invalid shot method: {shot_method}")

    if model_name == "text-davinci-003":
        text = "Q: "
        text += ",".join([str(x) for x in sequence])
        text += "\n"
        text += "Explanation: "
        text += fn
        text += "\n"
        text += "\n"
        return text

    elif model_name == "gpt-3.5-turbo":
        if base == 10:
            q_text = ",".join([str(x) for x in sequence])
        elif base == 2:
            q_text = ",".join([bin(x) for x in sequence])
        response = [{"role": "user", "content": q_text}]
        a_text = "Explanation: " + fn
        response += [{"role": "assistant", "content": a_text}]
        return response

    else:
        raise ValueError(f"Invalid model name: {model_name}")


def parse_explanation(model_response: str) -> tuple[str, str]:
    """
    Parse an explanation into a function.
    """
    # Splitting the string into lines
    lines = model_response.split("\n")

    # Initializing the variables with None
    x = ""

    # Looping over the lines
    for line in lines:
        # Splitting the line into key and value
        parts = line.split(": ", 1)
        if len(parts) == 2:
            key, value = parts
            # Saving the value based on the key
            if key == "Explanation":
                x = value
    return x
