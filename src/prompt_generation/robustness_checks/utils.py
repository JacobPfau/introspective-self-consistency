from typing import Dict, List, Optional, Tuple, Union

from src.evals.utils import generate_random_function, reformat_function
from src.models.openai_model import OpenAIChatModels, OpenAITextModels
from src.prompt_generation.prompt_loader import PromptBase, get_formatted_prompt
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
        fn, offset = generate_random_function(
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
    if base == 10:
        sequence_str = ",".join([str(x) for x in sequence])
    elif base == 2:
        sequence_str = ",".join([bin(x) for x in sequence])
    else:
        raise ValueError(f"Invalid base: {base}")
    if role_prompt is None:
        return text + get_formatted_prompt(
            PromptBase.ROLE_PROMPT,
            {"role_prompt": "", "seq": sequence_str, "base": base},
        )
    else:
        return text + get_formatted_prompt(
            PromptBase.ROLE_PROMPT,
            {
                "role_prompt": ROLE_PROMPTS[role_prompt],
                "seq": sequence_str,
                "base": base,
            },
        )
