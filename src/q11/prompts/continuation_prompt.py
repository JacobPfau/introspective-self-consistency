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
from typing import List

# from evals.utils import _generate_random_function, generate_wrong_functions
from models.openai_model import CHAT_MODEL_NAME, DAVINCI_MODEL_NAME

# from pipelines.sequence_completions import (  # BASE_PROMPT,; COT_PROMPT,; COT_STEP,
#     SYSTEM_PROMPT,
#     sequence_functions,
# )
from q11.prompts.distributions import DISTRIBUTIONS


def create_continuation_prompt(
    sequence: List[int],
    distribution: str,
    model_name: str = DAVINCI_MODEL_NAME,
    base: int = 10,
):
    """
    Create a prompt to continue a sequence of numbers.
    """

    text = DISTRIBUTIONS[distribution]["continuation"]
    text += "\n"
    text += f"The sequence is in base {base}."
    text += "\n"
    text += ",".join([str(x) for x in sequence])
    if model_name == DAVINCI_MODEL_NAME:
        text += "\n"
        text += "A: "
        return text
    elif model_name == CHAT_MODEL_NAME:
        return [{"role": "user", "content": text}]
    else:
        raise ValueError(f"Invalid model name: {model_name}")
