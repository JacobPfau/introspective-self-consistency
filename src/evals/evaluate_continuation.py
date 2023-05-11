from typing import List, Union

from src.models.openai_model import (
    DAVINCI_MODEL_NAME,
    CHAT_MODEL_NAME,
    generate_chat_completion,
    generate_completion,
)


def valid_continuation(
    model_continuation: str,
) -> bool:
    """
    Given a continuation as supplied by the model,
    return whether it is a valid integer or not.
    """
    try:
        # TODO: Work for arbitrary base continuation
        int(model_continuation)
    except ValueError:
        return False
    else:
        return True


def generate_continuation(
    prompt: Union[str, List[str]],
    model_name: str,
    temperature: int,
) -> str:
    """
    Given a prompt, generate a continuation from the model.
    """
    if model_name == "text-davinci-003":
        # Feed this into the model
        model_response = generate_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
            model=DAVINCI_MODEL_NAME,
        )
    elif model_name == "gpt-3.5-turbo":
        # Feed this into the model
        model_response = generate_chat_completion(
            prompt_turns=prompt,
            temperature=temperature,
            max_tokens=256,
            model=CHAT_MODEL_NAME,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model_response
