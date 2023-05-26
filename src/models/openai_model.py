import logging
import os
import time
from typing import Callable, List, Union

import openai

from src.models.utils import INVALID_RESPONSE, ExtendedEnum

CHAT_PROMPT_TEMPLATE = {"role": "user", "content": ""}
# TEXT_PROMPT_TEMPLATE is just a simple string or array of strings
DAVINCI_MODEL_NAME = "text-davinci-003"
CHAT_MODEL_NAME = "gpt-3.5-turbo"
_MAX_RETRIES = 3
_RETRY_TIMEOUT = 10
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


class OpenAITextModels(ExtendedEnum):
    TEXT_DAVINCI_003 = "text-davinci-003"
    DAVINCI = "davinci"


class OpenAIChatModels(ExtendedEnum):
    CHAT_GPT_35 = "gpt-3.5-turbo"
    CHAT_GPT_4 = "gpt-4-0314"


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_openai_model_from_string(model_name: str) -> ExtendedEnum:
    if model_name in OpenAITextModels.list():
        return OpenAITextModels(model_name)
    elif model_name in OpenAIChatModels.list():
        return OpenAIChatModels(model_name)
    else:
        raise KeyError(f"Invalid OpenAI model name: {model_name}")


def _with_retries(api_call: Callable) -> str:
    for _ in range(_MAX_RETRIES):
        try:
            return api_call()
        except openai.APIError:
            logger.warning("API Error. Sleep and try again.")
        except openai.error.RateLimitError:
            logger.error(
                "Rate limiting, Sleep and try again."
            )  # TBD: how long to wait?
        # TODO: may want to also handle ServiceUnavailableError, RateLimitError
        except KeyError:
            logger.warning("Unexpected response format. Sleep and try again.")
        finally:
            time.sleep(_RETRY_TIMEOUT)

    logger.error("Reached retry limit and did not obtain proper response")
    return INVALID_RESPONSE


def generate_completion(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    model: Union[str, OpenAITextModels] = OpenAITextModels.TEXT_DAVINCI_003,
) -> str:
    # docs: https://platform.openai.com/docs/api-reference/completions/create

    if isinstance(model, str):
        model = OpenAITextModels(model)

    def api_call():
        response = openai.Completion.create(
            model=model.value,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
        )
        return response["choices"][0]["text"]

    return _with_retries(api_call)


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

    return _with_retries(api_call)


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
        return generate_completion(
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
