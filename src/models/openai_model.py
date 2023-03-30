import os
from typing import List

import openai

DAVINCI_MODEL_NAME = "text-davinci-003"
CHAT_MODEL_NAME = "gpt-3.5-turbo"
# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_completion(
    prompt: str, temperature: int = 0, max_tokens: int = 256, model=DAVINCI_MODEL_NAME
) -> str:
    # docs: https://platform.openai.com/docs/api-reference/completions/create
    response = openai.Completion.create(
        model=model, prompt=prompt, temperature=temperature, max_tokens=max_tokens
    )

    if len(response["choices"]) == 0:
        raise KeyError("Response did not return enough `choices`")

    return response["choices"][0]["text"]


def generate_chat_completion(
    prompt_turns: List[dict],
    temperature: int = 0,
    max_tokens: int = 256,
    model=CHAT_MODEL_NAME,
) -> str:
    # docs: https://platform.openai.com/docs/api-reference/chat
    # TODO: may want to handle ServiceUnavailableError, RateLimitError
    response = openai.ChatCompletion.create(
        model=model,
        messages=prompt_turns,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if len(response["choices"]) == 0:
        raise KeyError("Response did not return enough `choices`")

    return response["choices"][0]["message"]["content"]


def generate_response_with_turns(
    model: str,
    turns: List[dict],
) -> str:
    """
    Helper function to generate a response given a list of turns.
    Routes to the appropriate model.
    Turns are collapsed into a single string for the davinci model.
    """
    if model == DAVINCI_MODEL_NAME:
        return generate_completion(
            prompt="\n".join([turn["content"] for turn in turns]),
            temperature=0,
            max_tokens=256,
        )
    elif model == CHAT_MODEL_NAME:
        return generate_chat_completion(
            prompt_turns=turns,
            temperature=0,
            max_tokens=256,
        )
    else:
        raise ValueError(f"Model {model} not supported")
