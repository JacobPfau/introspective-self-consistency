from logging import getLogger
from typing import Dict, List, Union

from src.models.openai_model import (
    OpenAIChatModels,
    OpenAITextModels,
    generate_chat_completion,
    generate_text_completion,
)

logger = getLogger(__name__)


def valid_explanation(
    fn_form: str,
    sequence_length: int,
) -> bool:
    """
    Given a function form and an offset as supplied by the model,
    return whether the string is a valid python function.
    """
    try:
        # TODO: need to have this work for an arbitrary number of arguments
        [eval(fn_form)(i) for i in range(sequence_length + 1)]
        return True
    except Exception as e:
        logger.info(e)
        return False


def generate_explanation(
    prompt: Union[str, List[Dict[str, str]]],
    model_name: str,
    temperature: float,
) -> str:
    """
    Given a prompt, generate an explanation from the model.
    TODO: refactor code, entirely copied from generate_continuation
    """
    if model_name in OpenAITextModels.list():
        assert isinstance(prompt, str)
        # Feed this into the model
        model_response = generate_text_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
            model=model_name,
        )
    elif model_name in OpenAIChatModels.list():
        assert isinstance(prompt, list)
        # Feed this into the model
        model_response = generate_chat_completion(
            prompt_turns=prompt,
            temperature=temperature,
            max_tokens=256,
            model=model_name,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    logger.debug("explain prompt: ", prompt)
    logger.debug("model_response: ", model_response)

    return model_response


def generate_implied_sequence(
    fn_form: str,
    sequence_length: int,
) -> List[int]:
    """
    Given a function form and an offset as supplied by the model,
    generate the sequence.
    """
    return [eval(fn_form)(i) for i in range(sequence_length)]


def generate_implied_continuation(
    fn_form: str,
    sequence_length: int,
) -> int:
    """
    Given a function form and an offset as supplied by the model,
    generate the next element of the sequence.
    """
    return eval(fn_form)(sequence_length)
