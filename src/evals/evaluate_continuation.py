from typing import Dict, List, Union

from src.models.openai_model import (
    OpenAIChatModels,
    OpenAITextModels,
    generate_chat_completion,
    generate_text_completion,
)


def valid_continuation(
    model_continuation: str,
    base: int,
) -> bool:
    """
    Given a continuation as supplied by the model,
    return whether it is a valid integer or not.
    If in biinary, the continuation will be prefixed with 0b.
    """
    try:
        int(model_continuation, base)
    except ValueError:
        return False
    else:
        return True


def generate_continuation(
    prompt: Union[str, List[Dict[str, str]]],
    model_name: str,
    temperature: float,
) -> str:
    """
    Given a prompt, generate a continuation from the model.
    """
    if model_name in OpenAITextModels.list():
        # Feed this into the model
        model_response = generate_text_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
            model=model_name,
        )
    elif model_name in OpenAIChatModels.list():
        # Feed this into the model
        model_response = generate_chat_completion(
            prompt_turns=prompt,
            temperature=temperature,
            max_tokens=256,
            model=model_name,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model_response
