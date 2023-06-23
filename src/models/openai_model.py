import logging
import os
import time
from typing import Callable, List, Tuple, TypeVar, Union

import openai

from src.models.base_model import BaseModel

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


def get_openai_model_from_string(model_name: str) -> BaseModel:
    if model_name in OpenAITextModels.list():
        return OpenAITextModels(model_name)
    elif model_name in OpenAIChatModels.list():
        return OpenAIChatModels(model_name)
    else:
        raise KeyError(f"Invalid OpenAI model name: {model_name}")


# introduce type variable to avoid circular imports
T = TypeVar("T")


def _with_retries(api_call: Callable[[], T], invalid_response: str) -> T:
    for _ in range(_MAX_RETRIES):
        try:
            return api_call()
        except openai.APIError:
            logger.warning("API Error. Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)
        except openai.error.RateLimitError:
            logger.error(
                "Rate limiting, Sleep and try again."
            )  # TBD: how long to wait?
            time.sleep(_RETRY_TIMEOUT)
        # TODO: may want to also handle ServiceUnavailableError, RateLimitError
        except KeyError:
            logger.warning("Unexpected response format. Sleep and try again.")
            time.sleep(_RETRY_TIMEOUT)

    logger.error("Reached retry limit and did not obtain proper response")
    return invalid_response


def _get_raw_text_model_response(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    model: OpenAITextModels = OpenAITextModels.TEXT_DAVINCI_003,
    logprobs: int = 0,
    echo: bool = False,
) -> Union[str, openai.Completion]:
    # docs: https://platform.openai.com/docs/api-reference/completions/create
    def api_call():
        return openai.Completion.create(
            model=model.value,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
            logprobs=logprobs,
            echo=echo,
        )

    return _with_retries(api_call, invalid_response=model.invalid_response)


def generate_text_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    model: Union[str, OpenAITextModels] = OpenAITextModels.TEXT_DAVINCI_003,
) -> str:
    if isinstance(model, str):
        model = OpenAITextModels(model)

    response = _get_raw_text_model_response(prompt, temperature, max_tokens, model)

    if len(response["choices"]) == 0:
        logger.error("Response did not return enough `choices`")
        return model.invalid_response
    elif response == model.invalid_response:
        return response

    return response["choices"][0]["text"]


def generate_text_completion_with_logprobs(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 0,
    model: Union[str, OpenAITextModels] = OpenAITextModels.TEXT_DAVINCI_003,
    logprobs: int = 5,
    echo: bool = True,
) -> Union[str, Tuple[List[str], List[float]]]:
    if isinstance(model, str):
        model = OpenAITextModels(model)

    response = _get_raw_text_model_response(
        prompt, temperature, max_tokens, model, logprobs=logprobs, echo=echo
    )

    if len(response["choices"]) == 0:
        logger.error("Response did not return enough `choices`")
        return model.invalid_response
    elif response == model.invalid_response:
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
    if isinstance(model, str):
        model = OpenAIChatModels(model)

    def api_call():
        response = openai.ChatCompletion.create(
            model=model.value,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=prompt_turns,
        )
        return response["choices"][0]["message"]["content"]

    return _with_retries(api_call, invalid_response=model.invalid_response)


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
        return model.invalid_response
    else:
        raise ValueError(f"Model {model} not supported")
