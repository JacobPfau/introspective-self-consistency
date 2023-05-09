"""
Create a prompt to continue a sequence of numbers, in an arbitrary base.

Prompts will take the form:
--------------------------
DISTRIBUTION_PROMPT.
Sequence

A:

--------------------------

The sequences will be taken from the list of ambiguous sequences.

"""

# import random
from typing import List, Union

from evals.utils import _generate_random_function

# from evals.utils import _generate_random_function, generate_wrong_functions
from models.openai_model import CHAT_MODEL_NAME, DAVINCI_MODEL_NAME
from pipelines.sequence_completions import sequence_functions
from q11.prompts.distributions import DISTRIBUTIONS

# from pipelines.sequence_completions import (  # BASE_PROMPT,; COT_PROMPT,; COT_STEP,
#     SYSTEM_PROMPT,
#     sequence_functions,
# )






def create_continuation_prompt(
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
    prompt_text = "" if model_name == DAVINCI_MODEL_NAME else []
    if shots > 0:
        for i in range(shots):
            # Note: we are using the sequence length implicitly specified by
            # the target sequence to generate the prompts.
            shot_prompt = generate_cont_shot_prompt(shot_method, sequence_length)
            prompt_text += shot_prompt

    # TODO: Need to fix!!

    text += DISTRIBUTIONS[distribution]["continuation"]
    text += "\n"
    text += f"The sequence is in base {base}."
    text += "\n"
    text += ",".join([str(x) for x in sequence])
    if model_name == DAVINCI_MODEL_NAME:
        # Prepend to the shots
        pretext = "Here are some examples of sequence continuations."
        pretext += "\n"
        text = pretext + text
        text += "\n"
        text += "A: "
        return text
    elif model_name == CHAT_MODEL_NAME:
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
    shot_method, sequence_length, model_name=DAVINCI_MODEL_NAME
):
    """
    Generate a single shot prompt for a continuation.
    """
    if shot_method == "random":
        fn, offset = _generate_random_function(sequence_functions, (0, 10), (0, 10))
        sequence = [eval(fn)(x + offset) for x in range(sequence_length)]
    else:
        raise ValueError(f"Invalid shot method: {shot_method}")

    if model_name == DAVINCI_MODEL_NAME:
        text = "Q: "
        text += ",".join([str(x) for x in sequence])
        text += "\n"
        text += "A: "
        text += str(eval(fn)(sequence_length + offset))
        text += "\n"
        return text

    elif model_name == CHAT_MODEL_NAME:
        q_text = ",".join([str(x) for x in sequence])
        response = [{"role": "user", "content": q_text}]
        a_text = str(eval(fn)(sequence_length + offset))
        response += [{"role": "assistant", "content": a_text}]
        return response

    else:
        raise ValueError(f"Invalid model name: {model_name}")
