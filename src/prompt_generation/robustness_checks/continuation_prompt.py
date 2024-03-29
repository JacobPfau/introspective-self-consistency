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

from src.models.openai_model import (
    DAVINCI_MODEL_NAME,
    OpenAIChatModels,
    OpenAITextModels,
)
from src.pipelines import (
    ShotSamplingType,
    get_binary_sequences_as_dict,
    get_sequences_as_dict,
)
from src.prompt_generation import PromptBase, get_formatted_prompt
from src.prompt_generation.robustness_checks.distribution_prompt import TASK_PROMPTS
from src.prompt_generation.robustness_checks.utils import (
    extend_prompt,
    generate_random_fn_sequence,
    initialise_prompt,
    start_question,
)

logger = logging.getLogger(__name__)


def create_continuation_prompt(
    sequence: List[int],
    task_prompt: str,
    model_name: str = DAVINCI_MODEL_NAME,
    base: int = 10,
    shots: int = 0,
    shot_method: ShotSamplingType = ShotSamplingType.RANDOM,
    role_prompt: Optional[str] = None,
    show_function_space: bool = False,
) -> Union[str, List[dict]]:
    """
    Create a prompt to continue a sequence of numbers.
    """

    if isinstance(shot_method, str):
        shot_method = ShotSamplingType(shot_method.lower())

    sequence_length = len(sequence)
    prompt_text = initialise_prompt(model_name)
    # Generate the few shot examples
    if shots > 0:
        for i in range(shots):
            # Note: we are using the sequence length implicitly specified by
            # the target sequence to generate the prompts.
            shot_prompt = generate_cont_shot_prompt(
                shot_method, sequence_length, model_name, base, i
            )
            prompt_text = extend_prompt(prompt_text, shot_prompt)

    # Generate the continuation prompt
    text = TASK_PROMPTS[task_prompt]["continuation"]
    text = start_question(text, sequence, base, role_prompt)
    # Combine together to form the final prompt
    if model_name in OpenAITextModels.list():
        # Prepend to the shots
        text = get_formatted_prompt(
            PromptBase.COMPLETION_SKELETON_TEXT,
            {"prompt_text": prompt_text, "text": text},
        )
        return text
    elif model_name in OpenAIChatModels.list():
        assert isinstance(prompt_text, list)
        if show_function_space:

            if base == 10:
                file_text = get_formatted_prompt(PromptBase.ROBUST_COMPLETION_BASE10)
                all_sequences = get_sequences_as_dict()
            elif base == 2:
                file_text = get_formatted_prompt(PromptBase.ROBUST_COMPLETION_BASE2)
                all_sequences = get_binary_sequences_as_dict()
            else:
                raise ValueError(f"Invalid base: {base}")

                # Add the functions to the pretext
            all_sequences_formatted = {
                sequence: all_sequences[sequence].format("a", "b")
                for sequence in all_sequences
            }

            for sequence_type in all_sequences_formatted:
                file_text += (
                    sequence_type + "->" + all_sequences_formatted[sequence_type] + "\n"
                )
            pretext = [
                {
                    "role": "system",
                    "content": file_text,
                }
            ]

        else:
            pretext = [
                {
                    "role": "system",
                    "content": "Here are some examples of sequence continuations.",
                }
            ]

        whole_prompt = (
            pretext
            + prompt_text
            + [{"role": "user", "content": text}]
            + [{"role": "assistant", "content": "A: "}]
        )
        return whole_prompt
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def generate_cont_shot_prompt(
    shot_method,
    sequence_length,
    model_name=DAVINCI_MODEL_NAME,
    base=10,
    shot=1,
    num_range=(0, 7),
    offset_range=(0, 7),
):
    """
    Generate a single shot prompt for a continuation.
    """
    if shot_method == ShotSamplingType.RANDOM:
        sequence_functions = get_sequences_as_dict()
        fn, sequence = generate_random_fn_sequence(
            sequence_functions, num_range, offset_range, sequence_length
        )
        for x in sequence:
            assert isinstance(x, int)
    else:
        raise ValueError(f"Invalid shot method: {shot_method}")

    if model_name in OpenAITextModels.list():
        if base == 10:
            sequence_str = ",".join([str(x) for x in sequence])
            continuation = str(eval(fn)(sequence_length))
        elif base == 2:
            sequence_str = ",".join([bin(x) for x in sequence])
            continuation = bin(eval(fn)(sequence_length))
        else:
            raise ValueError(f"Invalid base: {base}")
        text = get_formatted_prompt(
            PromptBase.COMPLETION_SHOT_TEXT,
            {"sequence": sequence_str, "continuation": continuation},
        )
        return text

    elif model_name in OpenAIChatModels.list():
        if base == 10:
            q_text = ",".join([str(x) for x in sequence])
            a_text = str(eval(fn)(sequence_length))
        elif base == 2:
            q_text = ",".join([bin(x) for x in sequence])
            a_text = bin(eval(fn)(sequence_length))
        else:
            raise ValueError(f"Invalid base: {base}")
        response = [{"role": "user", "content": q_text}]
        response += [{"role": "assistant", "content": a_text}]
        return response

    else:
        raise ValueError(f"Invalid model name: {model_name}")
