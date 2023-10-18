from typing import Dict, List, Optional, Tuple, Union

from src.evals.utils import _generate_random_function, reformat_function
from src.models.openai_model import OpenAIChatModels, OpenAITextModels
from src.prompt_generation.robustness_checks.distribution_prompt import ROLE_PROMPTS


def extend_prompt(
    prompt_text: Union[str, List[Dict[str, str]]],
    shot_text: Union[str, List[Dict[str, str]]],
) -> Union[str, List[Dict[str, str]]]:
    """
    Given a prompt, extend it with the shot text.
    """
    if isinstance(prompt_text, str):
        assert isinstance(shot_text, str)
        return shot_text + prompt_text
    elif isinstance(prompt_text, list):
        assert isinstance(shot_text, list)
        return shot_text + prompt_text
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt_text)}")


def generate_random_fn_sequence(
    sequence_functions: dict[str, str],
    num_range: Tuple[int, int],
    offset_range: Tuple[int, int],
    sequence_length: int,
    base: int = 10,
) -> Tuple[str, List[int]]:
    for _ in range(3):
        fn, offset = _generate_random_function(
            sequence_functions, num_range, offset_range
        )
        fn = reformat_function(fn, offset, base)
        try:
            sequence = [eval(fn)(x) for x in range(sequence_length)]
            for x in sequence:
                assert isinstance(x, int)
            return fn, sequence
        except RecursionError:
            pass
    raise ValueError("Kept generating improper recursive functions, try again!")


def initialise_prompt(
    model_name: str,
) -> Union[str, List[Dict[str, str]]]:
    """
    Given a model name, initialise the prompt.
    """

    if model_name in OpenAITextModels.list():
        return ""
    elif model_name in OpenAIChatModels.list():
        return []
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def start_question(
    text: str,
    sequence: List[int],
    base: int,
    role_prompt: Optional[str] = None,
) -> str:
    """
    Start the question to prompt the model with, using the role and sequence.
    """
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
    return text
