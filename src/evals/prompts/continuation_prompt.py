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

The sequences will be taken from the list of ambiguous sequences.

"""

import logging
from typing import List, Optional, Union

from src.evals.prompts.distribution_prompt import ROLE_PROMPTS, TASK_PROMPTS
from src.evals.utils import _generate_random_function, reformat_function
from src.models.openai_model import (
    DAVINCI_MODEL_NAME,
    OpenAIChatModels,
    OpenAITextModels,
)
from src.pipelines.sequence_completions import sequence_functions

# TODO: fix generating functions to include recursive progressions, an ok fix for now.
del sequence_functions["recursive_progression"]

logger = logging.getLogger(__name__)


def create_continuation_prompt(
    sequence: List[int],
    task_prompt: str,
    model_name: str = DAVINCI_MODEL_NAME,
    base: int = 10,
    shots: int = 0,
    shot_method: str = "random",
    role_prompt: Optional[str] = None,
) -> Union[str, List[dict]]:
    """
    Create a prompt to continue a sequence of numbers.
    """
    sequence_length = len(sequence)
    prompt_text = "" if model_name in OpenAITextModels.list() else []
    if shots > 0:
        for i in range(shots):
            # Note: we are using the sequence length implicitly specified by
            # the target sequence to generate the prompts.
            shot_prompt = generate_cont_shot_prompt(
                shot_method, sequence_length, model_name, base, i
            )
            prompt_text += shot_prompt

    text = TASK_PROMPTS[task_prompt]["continuation"]
    text += "\n"
    # TODO: Decide if we want role prompt to go here
    if role_prompt is not None:
        text += ROLE_PROMPTS[role_prompt]
        text += "\n"
    text += f"The sequence is in base {base}."
    text += "\nQ: "
    if base == 10:
        text += ",".join([str(x) for x in sequence])
    elif base == 2:
        text += ",".join([bin(x) for x in sequence])
    else:
        raise ValueError(f"Invalid base: {base}")
    if model_name in OpenAITextModels.list():
        # Prepend to the shots
        pretext = "Here are some examples of sequence continuations."
        pretext += "\n"
        text = pretext + prompt_text + text
        text += "\n"
        text += "A: "
        return text
    elif model_name in OpenAIChatModels.list():
        pretext = [
            {
                "role": "system",
                "content": "Here are some examples of sequence continuations.",
            }
        ]

        whole_prompt = pretext + prompt_text + [{"role": "user", "content": text}] + [{"role": "assistant", "content": "A: "}]
        return whole_prompt
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def generate_cont_shot_prompt(
    shot_method, sequence_length, model_name=DAVINCI_MODEL_NAME, base=10, shot=1
):
    """
    Generate a single shot prompt for a continuation.
    """
    if shot_method == "random":
        fn, offset = _generate_random_function(sequence_functions, (0, 7), (0, 7))
        # replace
        fn = reformat_function(fn, offset)
        sequence = [eval(fn)(x) for x in range(sequence_length)]
    else:
        raise ValueError(f"Invalid shot method: {shot_method}")
    if model_name in OpenAITextModels.list():
        text = "Q: "
        if base == 10:
            text += ",".join([str(x) for x in sequence])
            a_text = str(eval(fn)(sequence_length))
        elif base == 2:
            text += ",".join([bin(x) for x in sequence])
            a_text = bin(eval(fn)(sequence_length))
        text += "\n"
        text += "A: "
        text += a_text
        text += "\n"
        return text

    elif model_name in OpenAIChatModels.list():
        if base == 10:
            q_text = ",".join([str(x) for x in sequence])
            a_text = str(eval(fn)(sequence_length))
        elif base == 2:
            q_text = ",".join([bin(x) for x in sequence])
            a_text = bin(eval(fn)(sequence_length))
        response = [{"role": "user", "content": q_text}]
        response += [{"role": "assistant", "content": a_text}]
        return response

    else:
        raise ValueError(f"Invalid model name: {model_name}")
