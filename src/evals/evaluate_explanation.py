from typing import Dict, List, Union

from src.models.openai_model import (
    CHAT_MODEL_NAME,
    DAVINCI_MODEL_NAME,
    OpenAIChatModels,
    OpenAITextModels,
    generate_chat_completion,
    generate_text_completion,
)


def valid_explanation(
    fn_form: str,
    offset: int,
    sequence_length: int,
) -> bool:
    """
    Given a function form and an offset as supplied by the model,
    return whether the string is a valid python function.
    """
    try:
        # TODO: need to have this work for an arbitrary number of arguments
        [eval(fn_form.format(i + offset)) for i in range(sequence_length + 1)]
    except SyntaxError:
        return False
    except NameError:
        return False
    except TypeError:
        return False
    except ValueError:
        return False
    else:
        return True


def correct_explanation(
    fn_form: str,
    offset: int,
    sequence_length: int,
    sequence: List[int],
) -> bool:
    """
    Given a function form and an offset as supplied by the model,
    return whether the function correctly generates the sequence.
    """
    return all(
        eval(fn_form.format(i + offset)) == sequence[i] for i in range(sequence_length)
    )


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
        # Feed this into the model
        model_response = generate_text_completion(
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
            model=DAVINCI_MODEL_NAME,
        )
    elif model_name in OpenAIChatModels.list():
        # Feed this into the model
        model_response = generate_chat_completion(
            prompt_turns=prompt,
            temperature=temperature,
            max_tokens=256,
            model=CHAT_MODEL_NAME,
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    print("explain prompt: ", prompt)
    print("model_response: ", model_response)

    return model_response


def generate_implied_continuation(
    fn_form: str,
    offset: int,
    sequence_length: int,
) -> int:
    """
    Given a function form and an offset as supplied by the model,
    generate the next element of the sequence.
    """
    return eval(fn_form)(offset + sequence_length)
