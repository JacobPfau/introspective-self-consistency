import logging
import os
<<<<<<< HEAD
=======
import time
>>>>>>> a5581ad (fix while loop)
from enum import Enum
from typing import List, Union

import openai

CHAT_PROMPT_TEMPLATE = {"role": "user", "content": ""}
# TEXT_PROMPT_TEMPLATE is just a simple string or array of strings
DAVINCI_MODEL_NAME = "text-davinci-003"
CHAT_MODEL_NAME = "gpt-3.5-turbo"
_MAX_RETRIES = 3
INVALID_RESPONSE = "INVALID_RESPONSE"
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAITextModels(Enum):
    TEXT_DAVINCI_003 = "text-davinci-003"


class OpenAIChatModels(Enum):
    CHAT_GPT_35 = "gpt-3.5-turbo"
    CHAT_GPT_4 = "gpt-4-0314"


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)


def generate_completion(
    prompt: str,
    temperature: int = 0,
    max_tokens: int = 256,
    model: Union[str, OpenAITextModels] = OpenAITextModels.TEXT_DAVINCI_003,
) -> str:
    # docs: https://platform.openai.com/docs/api-reference/completions/create

    if isinstance(model, str):
        model = OpenAITextModels(model)

    response = openai.Completion.create(
        model=model.value, prompt=prompt, temperature=temperature, max_tokens=max_tokens
    )

    if len(response["choices"]) == 0:
        raise KeyError("Response did not return enough `choices`")

    return response["choices"][0]["text"]


def generate_chat_completion(
    prompt_turns: List[dict],
    temperature: float = 0.0,
    max_tokens: int = 256,
    model: Union[str, OpenAIChatModels] = OpenAIChatModels.CHAT_GPT_35,
) -> str:
    # docs: https://platform.openai.com/docs/api-reference/chat
    # TODO: may want to handle ServiceUnavailableError, RateLimitError
    if isinstance(model, str):
        model = OpenAIChatModels(model)

    response = None
    n_retries = 0
    while n_retries < _MAX_RETRIES:
        try:
            response = openai.ChatCompletion.create(
                model=model.value,
                messages=prompt_turns,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.APIError:
            logger.warning("API Error. Sleep and try again.")
            n_retries += 1
            time.sleep(3)

    if response is None and n_retries == _MAX_RETRIES:
        logger.error("Reached retry limit and did not obtain proper response")
        return INVALID_RESPONSE

    if len(response["choices"]) == 0:
        logger.error("Response did not return enough `choices`")
        return INVALID_RESPONSE

    return response["choices"][0]["message"]["content"]


def generate_response_with_turns(
    model: str,
    turns: List[dict],
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    """
    Helper function to generate a response given a list of turns.
    Routes to the appropriate model.
    Turns are collapsed into a single string for non-chat model.
    """
    if model in OpenAITextModels.list():
        return generate_text_completion(
            prompt="\n".join([turn["content"] for turn in turns]),
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    elif model in OpenAIChatModels.list():
        return generate_chat_completion(
            prompt_turns=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    else:
        raise ValueError(f"Model {model} not supported")
