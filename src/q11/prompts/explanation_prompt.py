"""
Create a prompt to continue a sequence of numbers, in an arbitrary base.

Prompts will take the form:
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

"""


from typing import List, Union

from evals.utils import _generate_random_function

# from evals.utils import _generate_random_function, generate_wrong_functions
from models.openai_model import DAVINCI_MODEL_NAME
from pipelines.sequence_completions import sequence_functions
from q11.prompts.distributions import DISTRIBUTIONS


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
    prompt_text = "" if model_name == "DAVINCI" else []
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
    else:
        raise ValueError(f"Invalid base: {base}")
    pre_prompt = """Here are some examples of sequence explanations, i.e. python functions
    which could have generated the preceding sequences, with associated offset."""
    if model_name == "DAVINCI":
        # Prepend to the shots
        pretext = pre_prompt + "\n"
        pretext += "\n"
        text = pretext + prompt_text + text
        text += "\n"
        return text
    elif model_name == "CHAT":
        pretext = [
            {
                "role": "system",
                "content": pre_prompt,
            }
        ]
        whole_prompt = pretext + prompt_text + [{"role": "user", "content": text}]
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
        sequence = [eval(fn)(x + offset) for x in range(sequence_length)]
    else:
        raise ValueError(f"Invalid shot method: {shot_method}")

    if model_name == "DAVINCI":
        text = "Q: "
        if base == 10:
            text += ",".join([str(x) for x in sequence])
        elif base == 2:
            text += ",".join([bin(x) for x in sequence])
        text += "\n"
        text += "Explanation: "
        if base == 10:
            text += fn
        elif base == 2:
            # Need to convert the function to binary.
            # Current approach: replace all occurences of x with int(x, 2),
            # Except the first x (i.e. lambda x: int(x, 2) ** 2)
            # TODO: Implement this / decide how to.
            # fn = fn.replace("x", "int(x, 2)", 1)
            text += fn
        text += "\n"
        text += "Offset: "
        text += str(offset)
        text += "\n"
        return text

    elif model_name == "CHAT":
        if base == 10:
            q_text = ",".join([str(x) for x in sequence])
        elif base == 2:
            q_text = ",".join([bin(x) for x in sequence])
        response = [{"role": "user", "content": q_text}]
        if base == 10:
            fn_text = fn
        elif base == 2:
            # Need to convert the function to binary.
            # Current approach: replace all occurences of x with int(x, 2),
            # Except the first x (i.e. lambda x: int(x, 2) ** 2)
            # fn_text = fn.replace("x", "int(x, 2)", 1)
            # TODO: Implement this / decide how to.
            fn_text = fn
        a_text = "Explanation: " + fn_text
        a_text += "\n"
        a_text += "Offset: " + str(offset)
        response += [{"role": "assistant", "content": a_text}]
        return response

    else:
        raise ValueError(f"Invalid model name: {model_name}")


def parse_explanation(model_response: str) -> tuple[str, str]:
    """
    Parse an explanation into a function and offset.
    """
    # Splitting the string into lines
    lines = model_response.split("\n")

    # Initializing the variables with None
    x = ""
    y = ""

    # Looping over the lines
    for line in lines:
        # Splitting the line into key and value
        parts = line.split(": ", 1)
        if len(parts) == 2:
            key, value = parts
            # Saving the value based on the key
            if key == "Explanation":
                x = value
            elif key == "Offset":
                y = value
    return x, y
