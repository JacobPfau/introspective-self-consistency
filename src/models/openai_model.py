import logging
import os
import time
from enum import Enum
from typing import List, Tuple, Union

import openai

from src.models.base_model import BaseModel
from src.models.utils import INVALID_RESPONSE

CHAT_PROMPT_TEMPLATE = {"role": "user", "content": ""}
# TEXT_PROMPT_TEMPLATE is just a simple string or array of strings
DAVINCI_MODEL_NAME = "text-davinci-003"
CHAT_MODEL_NAME = "gpt-3.5-turbo"
_MAX_RETRIES = 3
_RETRY_TIMEOUT = 10
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAITextModels(BaseModel):
    TEXT_DAVINCI_003 = "text-davinci-003"
    DAVINCI = "davinci"


class OpenAIChatModels(BaseModel):
    CHAT_GPT_35 = "gpt-3.5-turbo"
    CHAT_GPT_4 = "gpt-4-0314"


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_openai_model_from_string(model_name: str) -> Enum:
    if model_name in [m.value for m in OpenAITextModels]:
        return OpenAITextModels(model_name)
    elif model_name in [m.value for m in OpenAIChatModels]:
        return OpenAIChatModels(model_name)
    else:
        raise KeyError(f"Invalid OpenAI model name: {model_name}")


def _get_raw_text_model_response(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    model: Union[str, OpenAITextModels] = OpenAITextModels.TEXT_DAVINCI_003,
    logprobs: int = 0,
) -> str:
    # docs: https://platform.openai.com/docs/api-reference/completions/create

    if isinstance(model, str):
        model = OpenAITextModels(model)

    response = None

    for _ in range(_MAX_RETRIES):
        try:
            response = openai.Completion.create(
                model=model.value,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=logprobs,
            )
            return response
        except openai.APIError:
            logger.warning("API Error. Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)
        except openai.error.RateLimitError:
            logger.error("Rate limiting, Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)

    logger.error("Reached retry limit and did not obtain proper response")
    return INVALID_RESPONSE


def generate_text_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    model: Union[str, OpenAITextModels] = OpenAITextModels.TEXT_DAVINCI_003,
) -> str:
    response = _get_raw_text_model_response(prompt, temperature, max_tokens, model)

    if len(response["choices"]) == 0:
        logger.error("Response did not return enough `choices`")
        return INVALID_RESPONSE
    elif response == INVALID_RESPONSE:
        return response

    return response["choices"][0]["text"]


def generate_text_completion_with_logprobs(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 0,
    model: Union[str, OpenAITextModels] = OpenAITextModels.TEXT_DAVINCI_003,
    logprobs: int = 5,
    echo: bool = True,
) -> Tuple[List[str], List[float]]:

    response = openai.Completion.create(
        model=model.value,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
        echo=echo,
    )  # echo is required to get the previous tokens logprobs

    if len(response["choices"]) == 0:
        logger.error("Response did not return enough `choices`")
        return INVALID_RESPONSE
    elif response == INVALID_RESPONSE:
        return response

    logprob_dict = response["choices"][0]["logprobs"]
    # return a list of tokens and a list of logprobs because sub-tokens can appear multiple times
    return logprob_dict["tokens"], logprob_dict["token_logprobs"]


def generate_chat_completion(
    prompt_turns: List[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
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
            if response is not None:
                break
        except openai.APIError:
            logger.warning("API Error. Sleep and try again.")
            n_retries += 1
            time.sleep(_RETRY_TIMEOUT)

    if response is None:
        logger.error("Reached retry limit and did not obtain proper response")
        return INVALID_RESPONSE

    if len(response["choices"]) == 0:
        logger.error("Response did not return enough `choices`")
        return INVALID_RESPONSE

    return response["choices"][0]["message"]["content"]


def generate_response_with_turns(
    model: Union[str, BaseModel],
    turns: List[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Helper function to generate a response given a list of turns.
    Routes to the appropriate model.
    Turns are collapsed into a single string for non-chat model.
    """
    if model in OpenAITextModels.list() or isinstance(model, OpenAITextModels):
        return generate_text_completion(
            prompt="\n".join([turn["content"] for turn in turns]),
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    elif model in OpenAIChatModels.list() or isinstance(model, OpenAIChatModels):
        return generate_chat_completion(
            prompt_turns=turns,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
    else:
        raise ValueError(f"Model {model} not supported")


def generate_logprob_response_with_turns(
    model: Union[str, BaseModel],
    turns: List[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    logprobs: int = 5,
) -> Tuple[List[str], List[float]]:
    """
    Helper function to generate a response given a list of turns.
    Routes to the appropriate model.
    Turns are collapsed into a single string for non-chat model.
    """
    if model in OpenAITextModels.list() or isinstance(model, OpenAITextModels):
        return generate_text_completion_with_logprobs(
            prompt="\n".join([turn["content"] for turn in turns]),
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            logprobs=logprobs,
        )
    elif model in OpenAIChatModels.list() or isinstance(model, OpenAIChatModels):
        logger.error("Chat models don't support returning logprob")
        return INVALID_RESPONSE
    else:
        raise ValueError(f"Model {model} not supported")
